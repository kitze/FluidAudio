import Foundation
import XCTest

@testable import FluidAudio

/// Exercises production recovery planning, gap detection, and stitching. No copied algorithms,
/// model stubs, or synthetic audio. End-to-end recognition uses separate real-model measurements.
final class ChunkProcessorCoverageTests: XCTestCase {
    private let processor = ChunkProcessor(audioSamples: [])

    func testRecoveryChildrenOverlapAndShrinkWithoutDroppingOuterSamples() throws {
        let frame = ASRConstants.samplesPerEncoderFrame
        let parent = (100 * frame)..<(287 * frame + 733)
        let ranges = try XCTUnwrap(processor.recoveryRanges(for: parent))
        XCTAssertEqual(ranges.left.lowerBound, parent.lowerBound)
        XCTAssertEqual(ranges.right.upperBound, parent.upperBound)
        XCTAssertLessThan(ranges.right.lowerBound, ranges.left.upperBound)
        XCTAssertEqual(ranges.left.upperBound - ranges.right.lowerBound, 24 * frame)
        XCTAssertEqual(ranges.left.upperBound % frame, 0)
        XCTAssertEqual(ranges.right.lowerBound % frame, 0)
        XCTAssertLessThan(ranges.left.count, parent.count)
        XCTAssertLessThan(ranges.right.count, parent.count)
    }

    func testNestedRecoveryRemainsBoundedAndCoversBothOuterEdges() {
        var windows = [0..<239_360]
        for _ in 0..<3 {
            windows = windows.flatMap { window in
                guard let ranges = processor.recoveryRanges(for: window) else { return [window] }
                XCTAssertLessThan(ranges.left.count, window.count)
                XCTAssertLessThan(ranges.right.count, window.count)
                XCTAssertEqual(ranges.left.lowerBound, window.lowerBound)
                XCTAssertEqual(ranges.right.upperBound, window.upperBound)
                XCTAssertLessThan(ranges.right.lowerBound, ranges.left.upperBound)
                return [ranges.left, ranges.right]
            }
        }
        XCTAssertLessThanOrEqual(windows.count, 8)
        XCTAssertEqual(windows.first?.lowerBound, 0)
        XCTAssertEqual(windows.last?.upperBound, 239_360)
    }

    func testRecoveryRefusesSpansTooShortToSplit() {
        XCTAssertNil(processor.recoveryRanges(for: 0..<63_999))
    }

    func testInteriorGapsUseAbsoluteFramesAndClampToWindow() {
        let frame = ASRConstants.samplesPerEncoderFrame
        let gaps = ChunkProcessor.interiorGaps(
            timestamps: [90, 100, 114, 116, 140], window: (100 * frame)..<(135 * frame),
            samplesPerSecond: 16_000)
        XCTAssertEqual(gaps, [(100 * frame)..<(114 * frame), (116 * frame)..<(135 * frame)])
    }

    func testInteriorGapsIgnoreOrdinaryEmissionSpacingAndRepeatedFrames() {
        XCTAssertTrue(ChunkProcessor.interiorGaps(
            timestamps: [100, 100, 104, 110, 122], window: 0..<240_000,
            samplesPerSecond: 16_000).isEmpty)
        XCTAssertTrue(ChunkProcessor.interiorGaps(
            timestamps: [], window: 0..<240_000, samplesPerSecond: 16_000).isEmpty)
        XCTAssertTrue(ChunkProcessor.interiorGaps(
            timestamps: [100], window: 0..<240_000, samplesPerSecond: 16_000).isEmpty)
    }

    func testMatchedSeamKeepsIdenticalOverlapExactlyOnce() {
        let left: [ChunkProcessor.TokenWindow] = [(1, 100, 0.9), (2, 110, 0.9), (3, 120, 0.9)]
        let right: [ChunkProcessor.TokenWindow] = [(2, 111, 0.9), (3, 121, 0.9), (4, 130, 0.9)]
        XCTAssertEqual(processor.mergeChunksForTesting(left, right).map(\.token), [1, 2, 3, 4])
    }
}
