// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// @title LaplaceLedger
/// @notice An append-only record of hourly forecast commitments.
/// @dev Each issue commits a Merkle root over every forecast made at that hour, plus the hash of a manifest that says
/// how to verify it. A root can only be committed during the first `COMMIT_WINDOW` seconds of its issue hour, so the
/// block timestamp proves it was sealed before even the shortest (one-hour) outcome could be known. Issues only move
/// forward: nothing can be rewritten or backdated, and a missed hour stays visible as a gap. The committer is fixed
/// at deployment; rotating it means deploying a new ledger.
contract LaplaceLedger {
    /// @notice Seconds after the start of an issue hour during which its root can still be committed.
    uint256 public constant COMMIT_WINDOW = 15 minutes;

    /// @notice The only address allowed to commit.
    address public immutable committer;

    /// @notice The latest committed issue, as a unix hour (unix time / 3600); zero before the first commit.
    uint64 public lastIssue;

    /// @param issue The unix hour at whose start the forecasts were made.
    /// @param root Merkle root over every forecast leaf of the issue.
    /// @param manifest Hash of the issue's manifest (file locations, program hashes, feed version, data snapshot).
    /// @param leaves Number of leaves under the root.
    event Committed(uint64 indexed issue, bytes32 root, bytes32 manifest, uint32 leaves);

    error NotCommitter();
    error IssueNotAfterLast(uint64 issue, uint64 lastIssue);
    error OutsideCommitWindow(uint64 issue, uint256 timestamp);
    error EmptyCommitment();

    constructor(address committer_) {
        committer = committer_;
    }

    /// @notice Commit the forecasts of `issue`. Callable once per issue, in increasing order, inside its window.
    function commit(uint64 issue, bytes32 root, bytes32 manifest, uint32 leaves) external {
        if (msg.sender != committer) revert NotCommitter();
        if (issue <= lastIssue) revert IssueNotAfterLast(issue, lastIssue);
        uint256 opens = uint256(issue) * 1 hours;
        // The sequencer can shift block.timestamp by seconds at most, which is immaterial against a 15-minute window.
        // forge-lint: disable-next-line(block-timestamp)
        if (block.timestamp < opens || block.timestamp >= opens + COMMIT_WINDOW) {
            revert OutsideCommitWindow(issue, block.timestamp);
        }
        if (root == bytes32(0) || manifest == bytes32(0) || leaves == 0) revert EmptyCommitment();
        lastIssue = issue;
        emit Committed(issue, root, manifest, leaves);
    }
}
