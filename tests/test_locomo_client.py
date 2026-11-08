from unittest.mock import patch

import pytest

from neatmem.evaluation.locomo.client import NeatMemClient


@pytest.mark.parametrize("method_name", ["search", "search_with_graph"])
def test_search_allows_slow_server_side_rerank(method_name):
    with patch("neatmem.evaluation.locomo.client.requests.post") as post:
        post.return_value.json.return_value = {"results": []}
        client = NeatMemClient(base_url="http://localhost:8790")

        getattr(client, method_name)("question")

        post.assert_called_once()
        assert post.call_args.kwargs["timeout"] == 3000
