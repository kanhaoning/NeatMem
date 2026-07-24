"""Graph memory prompts and helpers — verbatim copy from mem0 1.0.11.

Sources (copied line-for-line, do not modify):
- EXTRACT_RELATIONS_PROMPT, DELETE_RELATIONS_SYSTEM_PROMPT, get_delete_messages:
  mem0/graphs/utils.py
- format_entities, remove_spaces_from_entities: mem0/memory/utils.py
- extract_json: mem0/memory/utils.py (used by the LLM response parser)

Any divergence in these strings changes LLM output and breaks equivalence
with mem0 1.0.11 (see plan section 4.5 / section 8 equivalence test).
"""

import re
from typing import Any, Dict, List


# ---- mem0/graphs/utils.py ------------------------------------------------

EXTRACT_RELATIONS_PROMPT = """

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.
CUSTOM_PROMPT

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user’s context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""


DELETE_RELATIONS_SYSTEM_PROMPT = """
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be deleted based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "USER_ID" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Deletion Criteria: Delete a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and delete as necessary.
   - Multiple deletions may be required based on the new information.
5. Semantic Integrity:
   - Ensure that deletions maintain or improve the overall semantic structure of the graph.
   - Avoid deleting relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only DELETE relationships that must be deleted and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT DELETE if their is a possibility of same type of relationship but different destination nodes.

For example:
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not delete in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
source -- relationship -- destination

Provide a list of deletion instructions, each specifying the relationship to be deleted.
"""


def get_delete_messages(existing_memories_string, data, user_id):
    return DELETE_RELATIONS_SYSTEM_PROMPT.replace(
        "USER_ID", user_id
    ), f"Here are the existing memories: {existing_memories_string} \n\n New Information: {data}"


# ---- mem0/memory/utils.py ------------------------------------------------

def format_entities(entities):
    if not entities:
        return ""

    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relationship']} -- {entity['destination']}"
        formatted_lines.append(simplified)

    return "\n".join(formatted_lines)


def remove_spaces_from_entities(
    entity_list: List[Any],
    *,
    sanitize_relationship: bool = True,
) -> List[Dict[str, Any]]:
    """Normalize entity relation dicts: lowercase, spaces to underscores.

    Copied verbatim from mem0/memory/utils.py. kuzu_memory.py calls this with
    sanitize_relationship=False (relationship stays a plain underscore string).
    """
    required = ("source", "relationship", "destination")
    cleaned: List[Dict[str, Any]] = []
    for item in entity_list:
        if not isinstance(item, dict) or not item:
            continue
        if not all(key in item for key in required):
            continue
        item["source"] = item["source"].lower().replace(" ", "_")
        rel = item["relationship"].lower().replace(" ", "_")
        # sanitize_relationship=False in the kuzu path (see kuzu_memory.py:657);
        # we mirror that call site. If sanitize_relationship were True we would
        # apply sanitize_relationship_for_cypher, but mem0's kuzu path does not.
        item["relationship"] = _sanitize_relationship_for_cypher(rel) if sanitize_relationship else rel
        item["destination"] = item["destination"].lower().replace(" ", "_")
        cleaned.append(item)
    return cleaned


def _sanitize_relationship_for_cypher(relationship) -> str:
    """Minimal stand-in for mem0's sanitizer (only reached if sanitize_relationship=True).

    mem0's kuzu path calls remove_spaces_from_entities with sanitize_relationship=False,
    so this is unused in practice but kept for API parity. It uppercases and replaces
    non-alphanumerics with underscores — a conservative default.
    """
    if not relationship:
        return ""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(relationship)).upper()


def extract_json(text):
    """Extract JSON content from a string — verbatim from mem0/memory/utils.py.

    Removes enclosing triple backticks (with optional 'json' tag). If no code
    block is found, locates JSON by first '{' and last '}'. Falls back to text.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
        else:
            json_str = text
    return json_str
