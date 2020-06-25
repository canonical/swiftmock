===============================================================================
swiftmock: A mock backend for testing swift.
===============================================================================


🐉 Installation
=================

Install from `PyPI`_:

  ::

    $ pip install swiftmock

.. _PyPI: https://www.pypi.org/project/swiftmock
.. _Github: https://github.com/canonical/swiftmock


🐉 Usage
==========

Importing directly
-------------------

You can use **swiftmock** directly in your code:

.. code:: python

    from swiftmock.swift import MockConnection
    with MockConnection() as conn:
        conn.put_container("fake-container")
        conn.put_object("fake-container", "path/to/object", b"contents")
        header, contents = conn.get_object("fake-container", "path/to/object")
    assert contents == b"contents"


Using with `Pytest`_
---------------------

You can also use this library as a **pytest** plugin.

.. code:: python

    def my_test_using_swift(mock_swift):
        # optional, the mock automatically replaces *swiftclient.client.Connection*
        # so that it automatically returns the mocked instance
        mock_swift.put_container("fake-container")
        with pytest.assert_raises(swiftclient.exceptions.ClientException):
            mock_swift.get_object("fake-container", "non/existent/object")


.. _Pytest: https://pytest.org


