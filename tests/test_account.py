def test_stat_account(test_cls):
    headers = test_cls.conn.head_account()
    test_cls.check_account_headers(headers)


def test_list_account(test_cls):
    headers, containers = test_cls.conn.get_account()
    test_cls.check_account_headers(headers)

    assert len(containers) > 0
    test_container = next(
        iter(c for c in containers if c.get("name") == test_cls.containername)
    )
    assert test_container.get("bytes") >= 0
    assert test_container.get("count") >= 0

    # Check if list limit is working
    headers, containers = test_cls.conn.get_account(limit=1)
    assert len(containers) == 1

    # Check full listing
    headers, containers = test_cls.conn.get_account(limit=1, full_listing=True)
    assert len(containers) >= 2  # there might be more containers

    # Test marker
    headers, containers = test_cls.conn.get_account(marker=test_cls.containername)
    assert len(containers) >= 1
    assert test_cls.containername_2 == containers[0].get("name")

    # Test prefix
    _, containers = test_cls.conn.get_account(prefix="dne")
    assert len(containers) == 0


def test_post_account(test_cls):
    test_cls.conn.post_account({"x-account-meta-data": "Something"})
    headers = test_cls.conn.head_account()
    assert headers.get("x-account-meta-data") == "Something"
