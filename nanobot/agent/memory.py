"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


@dataclass
class BlockSummary:
    """Mid-level block summary."""
    block_id: str
    start_index: int
    end_index: int
    timestamp_start: str
    timestamp_end: str
    message_count: int
    topics: list[str]
    key_points: list[str]
    decisions_outcomes: list[str]
    context_for_future: str


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


_SAVE_BLOCK_SUMMARY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_block_summary",
            "description": "Save a mid-level summary of this message block",
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-5 main topics discussed in this block"
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "5-10 key points with enough detail for future reference"
                    },
                    "decisions_outcomes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important decisions, actions taken, or outcomes (if any)"
                    },
                    "context_for_future": {
                        "type": "string",
                        "description": "1-2 sentences providing context for understanding future conversations"
                    }
                },
                "required": ["topics", "key_points", "context_for_future"]
            }
        }
    }
]


class MemoryStore:
    """Three-layer memory: MEMORY.md (long-term) + SUMMARIES.md (mid-level) + HISTORY.md (full log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.blocks_dir = ensure_dir(self.memory_dir / ".blocks")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.summaries_file = self.memory_dir / "SUMMARIES.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")


    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def get_summaries_context(self, session = None, max_blocks: int = 5) -> str:
        """
        Read recent block summaries for injection into System Prompt.

        Always loads the most recent 5 blocks (~125 messages) to provide
        consistent mid-level context without bloating the prompt.

        Args:
            session: Current session (unused, kept for API compatibility)
            max_blocks: Number of blocks to load (default: 5)

        Returns:
            Formatted summaries text
        """
        if not self.summaries_file.exists():
            return ""

        try:
            content = self.summaries_file.read_text(encoding="utf-8").strip()
            if not content:
                return ""

            # Load fixed number of recent blocks
            blocks = content.split("\n## Block ")
            if len(blocks) > max_blocks + 1:  # +1 for header
                blocks = [blocks[0]] + blocks[-(max_blocks):]
                content = "\n## Block ".join(blocks)

            return content
        except Exception as e:
            logger.warning("Failed to read summaries: {}", e)
            return ""


    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """
        Consolidate into MEMORY.md + HISTORY.md via LLM tool call.

        Strategy: Use block summaries when available and of sufficient quality.
        Falls back to raw messages if blocks are missing or low quality.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            keep_count = 0
            end_idx = len(session.messages)
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            end_idx = len(session.messages) - keep_count

        start_idx = session.last_consolidated
        messages_to_consolidate = session.messages[start_idx:end_idx]
        if not messages_to_consolidate:
            return True

        logger.info("Memory consolidation: {} to consolidate, {} keep", len(messages_to_consolidate), keep_count)

        # Try to use block summaries first
        all_blocks = self._load_block_metadata()
        blocks_to_consolidate = [
            b for b in all_blocks
            if b.start_index >= start_idx and b.end_index <= end_idx
        ]

        use_summaries = False
        if blocks_to_consolidate:
            # Quality check
            if self._blocks_have_sufficient_quality(blocks_to_consolidate):
                # Coverage check: blocks should cover most of the range
                covered = sum(b.message_count for b in blocks_to_consolidate)
                total = len(messages_to_consolidate)
                coverage = covered / total if total > 0 else 0

                if coverage >= 0.8:  # 80% coverage required
                    use_summaries = True
                    logger.info("Using block summaries for consolidation (coverage: {:.0%})", coverage)
                else:
                    logger.info("Block coverage insufficient ({:.0%}), using raw messages", coverage)
            else:
                logger.info("Block summaries quality insufficient, using raw messages")

        current_memory = self.read_long_term()

        if use_summaries:
            # Consolidate from block summaries (efficient)
            prompt = self._build_consolidation_prompt_from_summaries(
                current_memory, blocks_to_consolidate
            )
        else:
            # Consolidate from raw messages (fallback)
            prompt = self._build_consolidation_prompt_from_messages(
                current_memory, messages_to_consolidate
            )

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False

    def _blocks_have_sufficient_quality(self, blocks: list[BlockSummary]) -> bool:
        """Check if block summaries have enough detail for consolidation."""
        for block in blocks:
            # Must have at least 3 key points
            if len(block.key_points) < 3:
                logger.debug("Block {} has insufficient key_points: {}", block.block_id, len(block.key_points))
                return False
            # Topics and context cannot be empty
            if not block.topics or not block.context_for_future:
                logger.debug("Block {} missing topics or context", block.block_id)
                return False
        return True

    def _format_blocks_for_consolidation(self, blocks: list[BlockSummary]) -> str:
        """Format block summaries for consolidation prompt."""
        lines = []
        for block in blocks:
            block_num = block.block_id.split("_")[1]
            lines.append(f"### Block {block_num} | [{block.start_index}, {block.end_index}) | {block.timestamp_start[:10]}")
            lines.append(f"**Topics**: {', '.join(block.topics)}")
            if block.key_points:
                lines.append("**Key Points**:")
                for point in block.key_points:
                    lines.append(f"  - {point}")
            if block.decisions_outcomes:
                lines.append("**Decisions & Outcomes**:")
                for decision in block.decisions_outcomes:
                    lines.append(f"  - {decision}")
            lines.append(f"**Context**: {block.context_for_future}")
            lines.append("")
        return "\n".join(lines)

    def _build_consolidation_prompt_from_summaries(
        self,
        current_memory: str,
        blocks: list[BlockSummary]
    ) -> str:
        """Build consolidation prompt from block summaries (efficient path)."""
        blocks_text = self._format_blocks_for_consolidation(blocks)

        return f"""Process these conversation block summaries and call save_memory tool.

Extract and integrate:
- Important long-term facts and knowledge
- User preferences and patterns
- Key decisions and outcomes
- Project information and context

## Current Long-term Memory
{current_memory or "(empty)"}

## Block Summaries to Consolidate
{blocks_text}

Note: These are already-summarized blocks. Extract the most important long-term facts."""

    def _build_consolidation_prompt_from_messages(
        self,
        current_memory: str,
        messages: list[dict]
    ) -> str:
        """Build consolidation prompt from raw messages (fallback path)."""
        lines = []
        for m in messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        return f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""


    def _load_block_metadata(self) -> list[BlockSummary]:
        """Load all block summaries metadata from JSON files."""
        blocks = []
        for path in sorted(self.blocks_dir.glob("block_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                blocks.append(BlockSummary(**data))
            except Exception as e:
                logger.warning("Failed to load block {}: {}", path.name, e)
        return sorted(blocks, key=lambda b: b.start_index)

    def _save_block_metadata(self, block: BlockSummary) -> None:
        """Save block summary metadata to JSON."""
        path = self.blocks_dir / f"{block.block_id}.json"
        path.write_text(json.dumps(block.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """Format messages for summary generation."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # Simplify content
            if isinstance(content, list):
                # Multimodal: extract text only
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)

            # Truncate long content
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"[{timestamp[:19]}] {role}: {content}")

        return "\n".join(lines)

    def _archive_block_to_history(self, block: BlockSummary) -> None:
        """Archive block summary to HISTORY.md."""
        entry = (
            f"\n## {block.timestamp_start[:10]} | Block {block.block_id} | "
            f"Messages [{block.start_index}, {block.end_index})\n"
            f"**Topics**: {', '.join(block.topics)}\n"
            f"**Summary**: {block.context_for_future}\n"
        )

        # Append to HISTORY.md
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def _render_summaries_md(self, blocks: list[BlockSummary]) -> str:
        """Render block summaries as Markdown."""
        lines = [
            "# Recent Conversation Summaries",
            "",
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            ""
        ]

        for block in reversed(blocks):  # Most recent first
            block_num = block.block_id.split("_")[1]
            lines.append(f"## Block {block_num} | "
                        f"{block.timestamp_start[:16]} ~ {block.timestamp_end[11:16]} | "
                        f"{block.message_count} messages")
            lines.append("")

            if block.topics:
                lines.append("**Topics**: " + ", ".join(block.topics))
                lines.append("")

            if block.key_points:
                lines.append("**Key Points**:")
                for point in block.key_points:
                    lines.append(f"- {point}")
                lines.append("")

            if block.decisions_outcomes:
                lines.append("**Decisions & Outcomes**:")
                for item in block.decisions_outcomes:
                    lines.append(f"- {item}")
                lines.append("")

            if block.context_for_future:
                lines.append(f"**Context**: {block.context_for_future}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _update_summaries_file(self, max_blocks: int = 10) -> None:
        """Update SUMMARIES.md, keeping only the most recent max_blocks."""
        all_blocks = self._load_block_metadata()

        # Keep only recent blocks
        recent_blocks = all_blocks[-max_blocks:]

        # Clean up old block files
        if len(all_blocks) > max_blocks:
            for old_block in all_blocks[:-max_blocks]:
                old_path = self.blocks_dir / f"{old_block.block_id}.json"
                try:
                    # Archive to HISTORY.md
                    self._archive_block_to_history(old_block)
                    old_path.unlink()
                    logger.debug("Archived and deleted old block: {}", old_block.block_id)
                except Exception as e:
                    logger.warning("Failed to clean old block {}: {}", old_block.block_id, e)

        # Render SUMMARIES.md
        content = self._render_summaries_md(recent_blocks)
        self.summaries_file.write_text(content, encoding="utf-8")

    async def create_block_summary(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        block_size: int = 50,
        max_blocks_in_middle: int = 10
    ) -> bool:
        """
        Generate mid-level summary for unsummarized message block.

        Args:
            session: Current session
            provider: LLM provider
            model: Model name
            block_size: Messages per block
            max_blocks_in_middle: Max blocks to keep in middle layer

        Returns:
            True if a new block summary was created
        """
        # Calculate range of messages to summarize
        start_idx = session.last_block_summarized
        end_idx = len(session.messages)

        if end_idx - start_idx < block_size:
            return False  # Not enough messages

        # Take a full block
        block_end_idx = start_idx + block_size
        block_messages = session.messages[start_idx:block_end_idx]

        if not block_messages:
            return False

        logger.info("Creating block summary for messages [{}, {})", start_idx, block_end_idx)

        # Build LLM prompt
        messages_text = self._format_messages_for_summary(block_messages)

        prompt = f"""You are reviewing a block of {len(block_messages)} conversation messages.

Please analyze this conversation block and extract:
1. Main topics discussed (2-5 topics)
2. Key points with enough detail for future reference (5-10 points)
3. Important decisions, actions taken, or outcomes (if any)
4. Context needed for understanding future conversations (1-2 sentences)

Conversation block:
{messages_text}

Please call the save_block_summary tool with your analysis."""

        # Call LLM
        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a conversation summarization agent. Call the save_block_summary tool with your analysis."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_BLOCK_SUMMARY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Block summary: LLM did not call save_block_summary, skipping")
                return False

            # Parse tool call
            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Block summary: unexpected arguments type {}", type(args).__name__)
                return False

            # Create block summary
            block_id = f"block_{start_idx:06d}"
            block = BlockSummary(
                block_id=block_id,
                start_index=start_idx,
                end_index=block_end_idx,
                timestamp_start=block_messages[0].get("timestamp", datetime.now().isoformat()),
                timestamp_end=block_messages[-1].get("timestamp", datetime.now().isoformat()),
                message_count=len(block_messages),
                topics=args.get("topics", []),
                key_points=args.get("key_points", []),
                decisions_outcomes=args.get("decisions_outcomes", []),
                context_for_future=args.get("context_for_future", "")
            )

            # Save block metadata
            self._save_block_metadata(block)

            # Update SUMMARIES.md
            self._update_summaries_file(max_blocks_in_middle)

            # Update session's last_block_summarized
            session.last_block_summarized = block_end_idx

            logger.info("Created block summary: {} ({} messages)", block_id, block.message_count)
            return True

        except Exception as e:
            logger.exception("Failed to create block summary: {}", e)
            return False
