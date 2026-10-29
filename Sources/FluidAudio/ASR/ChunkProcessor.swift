import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]

    private let logger = AppLogger(category: "ChunkProcessor")
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

    /// Left context used by the single empty-window retry (4 encoder frames = 320ms).
    private let emptyWindowRetryContextSamples: Int = ASRConstants.samplesPerEncoderFrame * 4

    /// Mean absolute amplitude below which a chunk is treated as silence, so an empty window there
    /// is expected instead of suspicious.
    private let silenceAmplitudeThreshold: Float = 0.001

    private var maxModelSamples: Int { 240_000 }  // CoreML encoder capacity (15 seconds)
    private var chunkSamples: Int {
        // Reserve room for the retry context so chunkSamples + retryContext <= maxModelSamples
        let raw = max(
            maxModelSamples - emptyWindowRetryContextSamples - ASRConstants.melHopSize,
            ASRConstants.samplesPerEncoderFrame)
        return frameAligned(raw)
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
        var chunkDecoderState = TdtDecoderState.make()

        while chunkStart < audioSamples.count {
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= audioSamples.count
            let chunkEnd = isLastChunk ? audioSamples.count : candidateEnd

            if chunkEnd <= chunkStart {
                break
            }

            chunkDecoderState.reset()

            let chunkRange = chunkStart..<chunkEnd
            let chunkSamplesSlice = Array(audioSamples[chunkRange])

            var (windowTokens, windowTimestamps, windowConfidences) = try await transcribeChunk(
                samples: chunkSamplesSlice,
                contextSamples: 0,
                chunkStart: chunkStart,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &chunkDecoderState
            )

            // An empty window on audible audio means the decoder produced nothing at all for this
            // span, and its words are then missing from the merged transcript. Retry once with left
            // context from the previous window (the boundary-instability case), then carry on.
            if windowTokens.isEmpty && !isSilent(Array(audioSamples[chunkStart..<chunkEnd])),
                let retry = await retryEmptyWindow(
                    chunkStart: chunkStart, chunkEnd: chunkEnd, isLastChunk: isLastChunk, using: manager)
            {
                windowTokens = retry.tokens
                windowTimestamps = retry.timestamps
                windowConfidences = retry.confidences
            }

            // Combine tokens, timestamps, and confidences into aligned tuples
            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            let windowData: [TokenWindow] = zip(zip(windowTokens, windowTimestamps), windowConfidences).map {
                (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
            }
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

    /// Re-transcribes a window that decoded to nothing, feeding left context from the previous
    /// window. Returns nil when the retry cannot fit the encoder or also yields nothing.
    private func retryEmptyWindow(
        chunkStart: Int,
        chunkEnd: Int,
        isLastChunk: Bool,
        using manager: AsrManager
    ) async -> (tokens: [Int], timestamps: [Int], confidences: [Float])? {
        // The retry must still fit the encoder: window + context <= maxModelSamples.
        let windowLength = chunkEnd - chunkStart
        let contextBudget = max(maxModelSamples - windowLength, 0)
        let retryContextSamples = frameAligned(
            min(emptyWindowRetryContextSamples, contextBudget, chunkStart))
        guard retryContextSamples > 0 else { return nil }

        var retryDecoderState = TdtDecoderState.make()
        do {
            let retryResult = try await transcribeChunk(
                samples: Array(audioSamples[(chunkStart - retryContextSamples)..<chunkEnd]),
                contextSamples: retryContextSamples,
                chunkStart: chunkStart,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &retryDecoderState
            )
            return retryResult.0.isEmpty ? nil : retryResult
        } catch {
            // A failed retry must not fail the whole transcription; the window stays empty.
            logger.warning("Empty-window context retry failed at \(chunkStart) samples: \(error)")
            return nil
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
