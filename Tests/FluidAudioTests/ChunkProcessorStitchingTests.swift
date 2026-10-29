import Foundation
import XCTest

@testable import FluidAudio

/// Regression tests for long-audio chunk stitching.
///
/// Long clips (5+ minutes) lost whole spans of words at chunk boundaries. Three separate defects
/// contributed, and each has a test here:
/// 1. chunk/overlap/stride lengths were not encoder-frame aligned, so the integer global frame
///    offset rounded down and shifted a window's timestamps;
/// 2. an empty window result was silently accepted, so that span's words vanished;
/// 3. the fallback merge trimmed both sides around a midpoint, which drops right-window prefix
///    tokens and left-window tail tokens whenever the two windows disagree in the overlap.
final class ChunkProcessorStitchingTests: XCTestCase {

    private let sampleRate = 16000
    private let frameSamples = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms
    private let maxModelSamples = 240_000

    private func makeProcessor(totalSamples: Int = 16000 * 60) -> ChunkProcessor {
        ChunkProcessor(audioSamples: Array(repeating: 0.01, count: totalSamples))
    }

    // MARK: - Chunk plan

    func test_whenChunkPlanComputed_expectAllBoundariesFrameAligned() {
        let processor = makeProcessor()
        let ranges = processor.chunkRanges(forSampleCount: sampleRate * 600)

        XCTAssertGreaterThan(ranges.count, 40, "10 minutes should produce many windows")
        for range in ranges.dropLast() {
            XCTAssertEqual(
                range.lowerBound % frameSamples, 0,
                "chunk start \(range.lowerBound) must be encoder-frame aligned")
            XCTAssertGreaterThanOrEqual(
                range.count, frameSamples,
                "chunk length \(range.count) must be at least one encoder frame")
            XCTAssertEqual(
                range.count % frameSamples, 0,
                "non-final chunk length \(range.count) must be encoder-frame aligned")
        }
    }

    func test_whenChunkPlanComputed_expectStrideLeavesRealOverlap() {
        let processor = makeProcessor()
        let ranges = processor.chunkRanges(forSampleCount: sampleRate * 120)
        guard ranges.count > 2 else { return XCTFail("expected multiple windows") }

        let stride = ranges[1].lowerBound - ranges[0].lowerBound
        let windowLength = ranges[0].count
        let overlap = windowLength - stride

        XCTAssertEqual(stride % frameSamples, 0, "stride must be encoder-frame aligned")
        XCTAssertGreaterThanOrEqual(
            overlap, 2 * sampleRate,
            "each side must keep at least 2s of real overlap, got \(Double(overlap) / Double(sampleRate))s")
        XCTAssertLessThan(overlap, windowLength, "stride must advance the window")
    }

    func test_whenChunkPlanComputed_expectWindowsTileAudioWithoutGaps() {
        let processor = makeProcessor()
        let total = sampleRate * 300 + 733  // deliberately not frame aligned
        let ranges = processor.chunkRanges(forSampleCount: total)

        XCTAssertEqual(ranges.first?.lowerBound, 0)
        XCTAssertEqual(ranges.last?.upperBound, total, "the last window must reach the final sample")
        for (previous, next) in zip(ranges, ranges.dropFirst()) {
            XCTAssertLessThanOrEqual(
                next.lowerBound, previous.upperBound,
                "windows must overlap, never skip audio")
        }
    }

    func test_whenWindowReservesContext_expectModelLimitRespected() {
        let processor = makeProcessor()
        let ranges = processor.chunkRanges(forSampleCount: sampleRate * 60)
        guard let windowLength = ranges.first?.count else { return XCTFail("no windows") }

        // The window plus the retry's 4 frames of context must fit the 15s encoder capacity.
        XCTAssertLessThanOrEqual(windowLength + 4 * frameSamples, maxModelSamples)
    }

    func test_whenWindowsIterated_expectGlobalOffsetConversionIsExact() {
        let processor = makeProcessor()
        let ranges = processor.chunkRanges(forSampleCount: sampleRate * 90)

        for range in ranges {
            let frames = range.lowerBound / frameSamples
            XCTAssertEqual(
                frames * frameSamples, range.lowerBound,
                "global frame offset for \(range.lowerBound) must round-trip exactly")
        }
    }

    // MARK: - Merge

    func test_whenOverlapDisagrees_expectFallbackKeepsAllRightTokens() {
        // Windows whose overlap shares no token ids: the fallback path runs instead of LCS matching.
        // The old midpoint cutoff dropped the right window's prefix (token 90 at 4.8s), which is the
        // shape of the word loss seen on long clips.
        let left: [ChunkProcessor.TokenWindow] = [
            (1, 10, 0.9), (2, 30, 0.9), (3, 50, 0.9), (4, 70, 0.9),
        ]
        let right: [ChunkProcessor.TokenWindow] = [
            (90, 60, 0.9), (91, 80, 0.9), (92, 100, 0.9),
        ]

        let merged = ChunkProcessor(audioSamples: [0]).mergeChunksForTesting(left, right)
        let tokens = merged.map { $0.token }

        XCTAssertEqual(
            tokens, [1, 2, 3, 90, 91, 92],
            "left tokens before the right window and every right token must survive the merge")
    }

    func test_whenWindowsOverlapWithMatchingTokens_expectNoDuplicatesAtSeam() {
        // Four shared tokens in the 2s overlap region, then continuation tokens.
        let left: [ChunkProcessor.TokenWindow] = [
            (1, 100, 0.9), (2, 105, 0.9), (3, 110, 0.9), (4, 115, 0.9), (5, 120, 0.9),
        ]
        let right: [ChunkProcessor.TokenWindow] = [
            (3, 110, 0.9), (4, 115, 0.9), (5, 120, 0.9), (6, 125, 0.9), (7, 130, 0.9),
        ]

        let merged = ChunkProcessor(audioSamples: [0]).mergeChunksForTesting(left, right)
        let tokens = merged.map { $0.token }

        XCTAssertEqual(tokens, [1, 2, 3, 4, 5, 6, 7], "overlap tokens must not duplicate or vanish")
    }

    func test_whenLeftWindowEmpty_expectRightWindowKept() {
        let right: [ChunkProcessor.TokenWindow] = [(10, 50, 0.9), (11, 60, 0.9)]
        let merged = ChunkProcessor(audioSamples: [0]).mergeChunksForTesting([], right)
        XCTAssertEqual(merged.map { $0.token }, [10, 11])
    }

    func test_whenRightWindowEmpty_expectLeftWindowKept() {
        let left: [ChunkProcessor.TokenWindow] = [(10, 50, 0.9), (11, 60, 0.9)]
        let merged = ChunkProcessor(audioSamples: [0]).mergeChunksForTesting(left, [])
        XCTAssertEqual(merged.map { $0.token }, [10, 11])
    }
}
