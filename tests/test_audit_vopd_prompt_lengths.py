from scripts.audit_vopd_prompt_lengths import build_messages, distribution


def test_distribution_uses_nearest_rank():
    assert distribution(list(range(1, 101))) == {"p50": 50, "p95": 95, "p99": 99, "max": 100}


def test_build_messages_selects_requested_image_view():
    row = {
        "prompt": [{"role": "user", "content": "inspect <image> now"}],
        "images": [{"path": "/full.png"}],
        "bbox_images": [{"path": "/crop.png"}],
    }
    messages = build_messages(row, "bbox_images")
    assert messages[0]["content"] == [
        {"type": "text", "text": "inspect "},
        {"type": "image", "path": "/crop.png", "image": "/crop.png"},
        {"type": "text", "text": " now"},
    ]

