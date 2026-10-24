// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// @title LaplaceLedger
/// @notice An append-only record of Laplace's hourly forecasts. At the start of every hour the committer posts the
/// Merkle root of all forecasts made at that hour, within 15 minutes and so before any of them can settle. The root is
/// kept on-chain: any contract can check a forecast against it with `verifyForecast`.
/// @dev Leaves and proofs follow OpenZeppelin's `MerkleProof` conventions: a leaf is the double keccak of its ABI
/// encoding, and inner nodes hash the sorted pair of their children.
contract LaplaceLedger {
    uint256 public constant COMMIT_WINDOW = 15 minutes;

    address public immutable committer;
    uint64 public lastIssue;
    mapping(uint64 issue => bytes32 root) public rootOf;

    /// @param issue The forecast hour, in hours since the Unix epoch.
    /// @param root Merkle root over the issue's forecasts.
    /// @param manifest Hash of the manifest that defines the questions, forecasters and inputs.
    /// @param leaves Number of forecasts under the root.
    event Committed(uint64 indexed issue, bytes32 root, bytes32 manifest, uint32 leaves);

    error NotCommitter();
    error IssueNotAfterLast(uint64 issue, uint64 lastIssue);
    error OutsideCommitWindow(uint64 issue, uint256 timestamp);
    error EmptyCommitment();
    error ZeroCommitter();

    constructor(address committer_) {
        if (committer_ == address(0)) revert ZeroCommitter();
        committer = committer_;
    }

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
        rootOf[issue] = root;
        emit Committed(issue, root, manifest, leaves);
    }

    /// @notice The leaf of one forecast.
    /// @param forecaster keccak256 of the forecaster's definition (a program's source or a pool's specification).
    /// @param asset Index into the feed's append-only asset registry.
    /// @param horizon Hours ahead.
    /// @param bps Probabilities of the ten return bins in basis points, summing to 10,000.
    function leafHash(uint64 issue, bytes32 forecaster, uint8 asset, uint16 horizon, uint16[10] calldata bps)
        public
        pure
        returns (bytes32)
    {
        return keccak256(bytes.concat(keccak256(abi.encode(issue, forecaster, asset, horizon, bps))));
    }

    /// @notice Whether this forecast is in the record committed for `issue`.
    /// @param proof Sibling hashes from the leaf up to the root.
    function verifyForecast(
        uint64 issue,
        bytes32 forecaster,
        uint8 asset,
        uint16 horizon,
        uint16[10] calldata bps,
        bytes32[] calldata proof
    ) external view returns (bool) {
        bytes32 root = rootOf[issue];
        if (root == bytes32(0)) return false;
        bytes32 node = leafHash(issue, forecaster, asset, horizon, bps);
        for (uint256 i; i < proof.length; ++i) {
            bytes32 sibling = proof[i];
            node = node < sibling ? _hashPair(node, sibling) : _hashPair(sibling, node);
        }
        return node == root;
    }

    function _hashPair(bytes32 a, bytes32 b) private pure returns (bytes32) {
        return keccak256(abi.encode(a, b));
    }
}
