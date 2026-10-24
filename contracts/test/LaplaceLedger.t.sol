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
}
