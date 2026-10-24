// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {Test} from "forge-std/Test.sol";
import {LaplaceLedger} from "../src/LaplaceLedger.sol";

contract LaplaceLedgerTest is Test {
    LaplaceLedger internal ledger;
    address internal committer = makeAddr("committer");
    bytes32 internal constant ROOT = keccak256("root");
    bytes32 internal constant MANIFEST = keccak256("manifest");
    uint64 internal constant ISSUE = 497_983; // 2026-10-23 07:00 UTC as a unix hour

    function setUp() public {
        ledger = new LaplaceLedger(committer);
        vm.warp(uint256(ISSUE) * 1 hours + 2 minutes);
    }

    function test_CommitsInsideTheWindowAndEmits() public {
        vm.expectEmit(address(ledger));
        emit LaplaceLedger.Committed(ISSUE, ROOT, MANIFEST, 231);
        vm.prank(committer);
        ledger.commit(ISSUE, ROOT, MANIFEST, 231);
        assertEq(ledger.lastIssue(), ISSUE);
        assertEq(ledger.rootOf(ISSUE), ROOT);
    }

    function test_RequiresACommitter() public {
        vm.expectRevert(LaplaceLedger.ZeroCommitter.selector);
        new LaplaceLedger(address(0));
    }

    function test_RejectsAnyoneButTheCommitter(address caller) public {
        vm.assume(caller != committer);
        vm.prank(caller);
        vm.expectRevert(LaplaceLedger.NotCommitter.selector);
        ledger.commit(ISSUE, ROOT, MANIFEST, 231);
    }

    function test_IssuesOnlyMoveForward() public {
        vm.startPrank(committer);
        ledger.commit(ISSUE, ROOT, MANIFEST, 231);
        vm.expectRevert(abi.encodeWithSelector(LaplaceLedger.IssueNotAfterLast.selector, ISSUE, ISSUE));
        ledger.commit(ISSUE, ROOT, MANIFEST, 231);
        vm.warp(uint256(ISSUE + 3) * 1 hours + 1 minutes);
        ledger.commit(ISSUE + 3, ROOT, MANIFEST, 231); // a gap of missed hours is allowed and stays visible
        vm.expectRevert(abi.encodeWithSelector(LaplaceLedger.IssueNotAfterLast.selector, ISSUE + 2, ISSUE + 3));
        ledger.commit(ISSUE + 2, ROOT, MANIFEST, 231);
        vm.stopPrank();
        assertEq(ledger.rootOf(ISSUE + 2), bytes32(0));
    }

    function test_RejectsCommitsBeforeTheIssueHour() public {
        vm.warp(uint256(ISSUE) * 1 hours - 1);
        vm.prank(committer);
        vm.expectRevert(
            abi.encodeWithSelector(LaplaceLedger.OutsideCommitWindow.selector, ISSUE, uint256(ISSUE) * 1 hours - 1)
        );
        ledger.commit(ISSUE, ROOT, MANIFEST, 231);
    }

    function test_RejectsCommitsAfterTheWindow() public {
        uint256 late = uint256(ISSUE) * 1 hours + ledger.COMMIT_WINDOW();
        vm.warp(late);
        vm.prank(committer);
        vm.expectRevert(abi.encodeWithSelector(LaplaceLedger.OutsideCommitWindow.selector, ISSUE, late));
        ledger.commit(ISSUE, ROOT, MANIFEST, 231);
    }

    function test_RejectsEmptyCommitments() public {
        vm.startPrank(committer);
        vm.expectRevert(LaplaceLedger.EmptyCommitment.selector);
        ledger.commit(ISSUE, bytes32(0), MANIFEST, 231);
        vm.expectRevert(LaplaceLedger.EmptyCommitment.selector);
        ledger.commit(ISSUE, ROOT, bytes32(0), 231);
        vm.expectRevert(LaplaceLedger.EmptyCommitment.selector);
        ledger.commit(ISSUE, ROOT, MANIFEST, 0);
        vm.stopPrank();
    }

    /// Any accepted commit happened inside its issue's window and advanced the ledger.
    function testFuzz_AcceptedCommitsAreTimelyAndMonotonic(uint64 issue, uint256 offset) public {
        issue = uint64(bound(issue, 1, type(uint32).max));
        offset = bound(offset, 0, 2 hours);
        uint256 at = uint256(issue) * 1 hours + offset;
        vm.warp(at);
        vm.prank(committer);
        try ledger.commit(issue, ROOT, MANIFEST, 1) {
            assertLt(offset, ledger.COMMIT_WINDOW());
            assertEq(ledger.lastIssue(), issue);
        } catch {
            assertGe(offset, ledger.COMMIT_WINDOW());
        }
    }

    /// The engine's test vectors: its leaf hashes match `leafHash`, its proofs verify against its root, and a forecast
    /// changed by a single basis point does not.
    function test_VerifiesTheEngineVectors() public {
        string memory json = vm.readFile(string.concat(vm.projectRoot(), "/test/vectors/record.json"));
        uint64 issue = uint64(vm.parseJsonUint(json, ".issue"));
        uint256 count = vm.parseJsonUint(json, ".count");
        vm.warp(uint256(issue) * 1 hours + 1 minutes);
        vm.prank(committer);
        ledger.commit(issue, vm.parseJsonBytes32(json, ".root"), MANIFEST, uint32(count));

        for (uint256 i; i < count; ++i) {
            string memory at = string.concat(".leaves[", vm.toString(i), "]");
            bytes32 forecaster = vm.parseJsonBytes32(json, string.concat(at, ".forecaster"));
            uint8 asset = uint8(vm.parseJsonUint(json, string.concat(at, ".asset")));
            uint16 horizon = uint16(vm.parseJsonUint(json, string.concat(at, ".horizon")));
            uint16[10] memory bps = _bps(vm.parseJsonUintArray(json, string.concat(at, ".bps")));
            bytes32[] memory proof = vm.parseJsonBytes32Array(json, string.concat(at, ".proof"));

            assertEq(
                ledger.leafHash(issue, forecaster, asset, horizon, bps),
                vm.parseJsonBytes32(json, string.concat(at, ".hash"))
            );
            assertTrue(ledger.verifyForecast(issue, forecaster, asset, horizon, bps, proof));
            assertFalse(ledger.verifyForecast(issue + 1, forecaster, asset, horizon, bps, proof));
            assertFalse(ledger.verifyForecast(issue, forecaster, asset, horizon == 1 ? 4 : 1, bps, proof));
            (bps[4], bps[5]) = (bps[4] + 1, bps[5] - 1);
            assertFalse(ledger.verifyForecast(issue, forecaster, asset, horizon, bps, proof));
        }
    }

    function test_ASingleForecastIsItsOwnRoot() public {
        uint16[10] memory bps = [uint16(1), 99, 900, 2000, 2000, 2000, 2000, 900, 99, 1];
        bytes32 leaf = ledger.leafHash(ISSUE, keccak256("program"), 0, 24, bps);
        vm.prank(committer);
        ledger.commit(ISSUE, leaf, MANIFEST, 1);
        assertTrue(ledger.verifyForecast(ISSUE, keccak256("program"), 0, 24, bps, new bytes32[](0)));
    }

    function test_NothingVerifiesForAnUncommittedIssue() public view {
        uint16[10] memory bps;
        bytes32[] memory proof = new bytes32[](0);
        assertFalse(ledger.verifyForecast(ISSUE, bytes32(0), 0, 1, bps, proof));
    }

    function _bps(uint256[] memory raw) private pure returns (uint16[10] memory out) {
        assertEq(raw.length, 10);
        for (uint256 j; j < 10; ++j) {
            out[j] = uint16(raw[j]);
        }
    }
}
