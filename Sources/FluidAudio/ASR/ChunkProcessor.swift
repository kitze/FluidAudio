import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]

    typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float)
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.9s of audio per window to stay under the encoder limit
    // - 2.0s overlap to give the decoder slack when merging windows
    //
    // Windows, overlap and stride are all whole numbers of encoder frames. The decoder reports
    // timestamps in encoder frames and the merge compares windows by those timestamps, while the
    // global offset of a window is chunkStart / samplesPerEncoderFrame (integer division). With
    // unaligned lengths that division truncates, every window is placed up to a frame early, and
    // the overlap merge trims the wrong side of the seam: on long clips that silently drops whole
    // spans of words (13-word and longer deletions on a 15 minute LibriSpeech clip).
    private let sampleRate: Int = 16000
    private let overlapSeconds: Double = 2.0

    /// A window whose first token starts later than this after the window start has lost its head.
    private static let headGapThresholdSeconds: Double = 1.0

    /// A last window whose last token ends earlier than this before the window end lost its tail.
    private static let tailGapThresholdSeconds: Double = 1.0

    /// Recovery splits a bad window in half; stop splitting below this span.
    private static let minimumRecoverySpanSamples: Int = 32_000  // 2s

    /// How many times a span may be halved while recovering.
    private static let maximumRecoveryDepth: Int = 3

    /// Mean absolute amplitude below which a span is treated as silence, so an empty decode there
    /// is expected instead of suspicious.
    private let silenceAmplitudeThreshold: Float = 0.001

    private var maxModelSamples: Int { 240_000 }  // CoreML encoder capacity (15 seconds)
    private var chunkSamples: Int {
        // Largest frame-aligned window that fits the encoder capacity (187 frames = 14.96s).
        frameAligned(maxModelSamples - ASRConstants.melHopSize)
    }
    private var overlapSamples: Int {
        let requested = Int(overlapSeconds * Double(sampleRate))
        return frameAligned(min(requested, chunkSamples / 2))
    }
    private var strideSamples: Int {
        frameAligned(max(chunkSamples - overlapSamples, ASRConstants.samplesPerEncoderFrame))
    }

    private func frameAligned(_ samples: Int) -> Int {
        max(samples / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame, 0)
    }

    /// Inclusive-exclusive sample ranges this processor will transcribe for a given audio length.
    /// Exposed for tests: the ranges must tile the audio without gaps and end exactly at the last sample.
    func chunkRanges(forSampleCount totalSamples: Int) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var chunkStart = 0
        while chunkStart < totalSamples {
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= totalSamples
            let chunkEnd = isLastChunk ? totalSamples : candidateEnd
            if chunkEnd <= chunkStart { break }
            ranges.append(chunkStart..<chunkEnd)
            if isLastChunk { break }
            chunkStart += strideSamples
        }
        return ranges
    }

    func process(
        using manager: AsrManager, startTime: Date
    ) async throws -> ASRResult {
        var chunkOutputs: [[TokenWindow]] = []

        var chunkStart = 0

        while chunkStart < audioSamples.count {
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= audioSamples.count
            let chunkEnd = isLastChunk ? audioSamples.count : candidateEnd

            if chunkEnd <= chunkStart {
                break
            }

            let windowData = try await transcribeWindow(
                chunkStart: chunkStart,
                chunkEnd: chunkEnd,
                isLastChunk: isLastChunk,
                using: manager
            )

            chunkOutputs.append(windowData)

            if isLastChunk {
                break
            }

            chunkStart += strideSamples
        }

        guard var mergedTokens = chunkOutputs.first else {
            return manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if chunkOutputs.count > 1 {
            for chunk in chunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(mergedTokens, chunk)
            }
        }

        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }

        return manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    /// True when a window's tokens do not cover the window and the missing audio is not silence.
    ///
    /// Deterministic blank decodes are real: the same span decodes to nothing at one window length
    /// and fine at another (14.64s window: 0 tokens; 12.00s / 14.99s window over the same audio:
    /// full transcript). Without this check such a window contributes nothing to the merge and its
    /// ~10s of speech is silently deleted from the transcript.
    static func needsRecovery(
        tokenCount: Int,
        firstTokenStart: Int?,
        lastTokenEnd: Int?,
        windowStart: Int,
        windowEnd: Int,
        isLastWindow: Bool,
        hasAudibleAudio: Bool,
        samplesPerSecond: Int,
        headGapThresholdSeconds: Double = ChunkProcessor.headGapThresholdSeconds,
        tailGapThresholdSeconds: Double = ChunkProcessor.tailGapThresholdSeconds
    ) -> Bool {
        guard hasAudibleAudio else { return false }
        if tokenCount == 0 || firstTokenStart == nil {
            return windowEnd - windowStart >= ChunkProcessor.minimumRecoverySpanSamples
        }
        let headGap = Double((firstTokenStart ?? windowStart) - windowStart) / Double(samplesPerSecond)
        if headGap > headGapThresholdSeconds {
            return true
        }
        guard isLastWindow, let lastEnd = lastTokenEnd else { return false }
        let tailGap = Double(windowEnd - lastEnd) / Double(samplesPerSecond)
        return tailGap > tailGapThresholdSeconds
    }

    /// Transcribes one window, and if the result does not cover the window it halves the span and
    /// transcribes the halves instead, until the span is covered or it is too short to split.
    ///
    /// Splitting changes the window length, which is what recovers a span the decoder returned
    /// nothing for at the original length. Timestamps stay absolute, so the halves merge normally.
    private func transcribeWindow(
        chunkStart: Int,
        chunkEnd: Int,
        isLastChunk: Bool,
        depth: Int = 0,
        using manager: AsrManager
    ) async throws -> [TokenWindow] {
        let windowData = try await transcribeSpan(
            start: chunkStart,
            end: chunkEnd,
            contextSamples: 0,
            isLastChunk: isLastChunk,
            using: manager
        )

        let audible = !isSilent(Array(audioSamples[chunkStart..<chunkEnd]))
        let firstStart = windowData.first.map {
            Int(Double($0.timestamp) * Double(ASRConstants.samplesPerEncoderFrame))
        }
        let lastEnd = windowData.last.map {
            Int(Double($0.timestamp + 1) * Double(ASRConstants.samplesPerEncoderFrame))
        }
        let interiorGap = Self.interiorGaps(
            timestamps: windowData.map(\.timestamp), window: chunkStart..<chunkEnd,
            samplesPerSecond: sampleRate
        ).contains { !isSilent(Array(audioSamples[$0])) }
        let recover = interiorGap || Self.needsRecovery(
            tokenCount: windowData.count,
            firstTokenStart: firstStart,
            lastTokenEnd: lastEnd,
            windowStart: chunkStart,
            windowEnd: chunkEnd,
            isLastWindow: isLastChunk,
            hasAudibleAudio: audible,
            samplesPerSecond: sampleRate
        )
        guard recover,
            depth < Self.maximumRecoveryDepth,
            chunkEnd - chunkStart >= Self.minimumRecoverySpanSamples * 2
        else {
            return windowData
        }

        guard let ranges = recoveryRanges(for: chunkStart..<chunkEnd) else { return windowData }
        let left = try await transcribeWindow(
            chunkStart: ranges.left.lowerBound, chunkEnd: ranges.left.upperBound,
            isLastChunk: false, depth: depth + 1, using: manager)
        let right = try await transcribeWindow(
            chunkStart: ranges.right.lowerBound, chunkEnd: ranges.right.upperBound,
            isLastChunk: isLastChunk, depth: depth + 1, using: manager)

        // Compare after stitching so overlap duplicates cannot inflate the apparent recovery.
        // A split that adds no tokens must not replace what the original window already decoded.
        let recovered = mergeChunks(left, right)
        guard recovered.count > windowData.count else { return windowData }
        return recovered
    }

    /// Long inter-emission gaps are suspicious only when their audio is audible. Callers perform
    /// that acoustic check separately; silence and long pauses must not cause recovery by themselves.
    static func interiorGaps(
        timestamps: [Int], window: Range<Int>, samplesPerSecond: Int,
        thresholdSeconds: Double = ChunkProcessor.headGapThresholdSeconds
    ) -> [Range<Int>] {
        zip(timestamps, timestamps.dropFirst()).compactMap { left, right in
            let start = max(window.lowerBound, left * ASRConstants.samplesPerEncoderFrame)
            let end = min(window.upperBound, right * ASRConstants.samplesPerEncoderFrame)
            guard end > start, Double(end - start) / Double(samplesPerSecond) > thresholdSeconds else {
                return nil
            }
            return start..<end
        }
    }

    /// Recovery windows need the same acoustic overlap as primary windows. Both children must
    /// shrink so recursion stays bounded, while the outer boundaries cover every original sample.
    func recoveryRanges(for window: Range<Int>) -> (left: Range<Int>, right: Range<Int>)? {
        guard window.count >= Self.minimumRecoverySpanSamples * 2 else { return nil }
        let mid = frameAligned(window.lowerBound + window.count / 2)
        let halfOverlap = frameAligned(min(overlapSamples / 2, window.count / 4))
        let leftEnd = min(mid + halfOverlap, window.upperBound)
        let rightStart = max(mid - halfOverlap, window.lowerBound)
        guard leftEnd > window.lowerBound, leftEnd < window.upperBound,
            rightStart > window.lowerBound, rightStart < window.upperBound else { return nil }
        return (window.lowerBound..<leftEnd, rightStart..<window.upperBound)
    }

    /// Transcribes `[start, end)` with optional prepended context, in absolute timestamps.
    private func transcribeSpan(
        start: Int,
        end: Int,
        contextSamples: Int,
        isLastChunk: Bool,
        using manager: AsrManager
    ) async throws -> [TokenWindow] {
        let context = min(contextSamples, start)
        let contextStart = start - context
        guard end > contextStart else { return [] }

        var decoderState = TdtDecoderState.make()
        let (tokens, timestamps, confidences) = try await transcribeChunk(
            samples: Array(audioSamples[contextStart..<end]),
            contextSamples: context,
            chunkStart: start,
            isLastChunk: isLastChunk,
            using: manager,
            decoderState: &decoderState
        )
        guard tokens.count == timestamps.count, tokens.count == confidences.count else {
            throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
        }
        return zip(zip(tokens, timestamps), confidences).map {
            (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
        }
    }

    private func transcribeChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float]) {
        guard !samples.isEmpty else { return ([], [], []) }

        // Defensive: never exceed the encoder capacity, even if a caller (e.g. the empty-window
        // retry) hands in a window plus context over the limit. Keeping the prefix preserves the
        // prepended context and the start of the window.
        let windowSamples = samples.count > maxModelSamples ? Array(samples.prefix(maxModelSamples)) : samples
        let contextSamples = min(contextSamples, windowSamples.count)

        let paddedChunk = manager.padAudioIfNeeded(windowSamples, targetLength: maxModelSamples)

        // Frame count covers the real window only; the prepended context is skipped by the decoder.
        let actualAudioSamples = max(windowSamples.count - contextSamples, 0)
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)
        // chunkStart is frame-aligned, so this offset is exact rather than truncated.
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame
        let contextFrames = contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: windowSamples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences)
    }

    /// Exposed for the stitching regression tests.
    func mergeChunksForTesting(_ left: [TokenWindow], _ right: [TokenWindow]) -> [TokenWindow] {
        mergeChunks(left, right)
    }

    private func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow]
    ) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = Double(ASRConstants.samplesPerEncoderFrame) / Double(sampleRate)
        let overlapDuration = overlapSeconds
        let halfOverlapWindow = overlapDuration / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        if leftEndTime <= rightStartTime {
            return left + right
        }

        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeAtRightStart(left: left, right: right, frameDuration: frameDuration)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        let contiguousPairs = findBestContiguousPairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right
            )
        }

        let lcsPairs = findLongestCommonSubsequencePairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        guard !lcsPairs.isEmpty else {
            return mergeAtRightStart(left: left, right: right, frameDuration: frameDuration)
        }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }

    private func findBestContiguousPairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                let leftToken = overlapLeft[i]
                let rightToken = overlapRight[j]

                if tokensMatch(leftToken, rightToken, tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < overlapLeft.count && l < overlapRight.count {
                        let nextLeft = overlapLeft[k]
                        let nextRight = overlapRight[l]

                        if tokensMatch(nextLeft, nextRight, tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private func findLongestCommonSubsequencePairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return pairs.reversed()
    }

    private func tokensMatch(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }

    /// Fallback merge when the two windows share no matchable tokens.
    ///
    /// This never discards right-hand tokens: the right window re-covers the overlap and continues
    /// past it, so its content is authoritative from its first token onward. Left tokens that end
    /// after the right window's first token are overlap duplicates and are dropped; everything else
    /// is kept. A midpoint cutoff instead drops right-side prefix tokens and left-side tail tokens,
    /// which silently loses words whenever the two windows disagree in the overlap.
    private func mergeAtRightStart(
        left: [TokenWindow],
        right: [TokenWindow],
        frameDuration: Double
    ) -> [TokenWindow] {
        guard let firstRight = right.first else { return left }
        let rightStartTime = Double(firstRight.timestamp) * frameDuration
        let keptLeft = left.filter { Double($0.timestamp) * frameDuration < rightStartTime }
        return keptLeft + right
    }

    /// Mean absolute amplitude below the silence threshold means an empty window is expected.
    private func isSilent(_ samples: [Float]) -> Bool {
        guard !samples.isEmpty else { return true }
        var sum: Float = 0
        for sample in samples {
            sum += abs(sample)
        }
        return sum / Float(samples.count) < silenceAmplitudeThreshold
    }
}
