"""Optional REAL PostgreSQL concurrency test.

Run with GATEKEEPER_TEST_POSTGRES_URL=postgresql+asyncpg://... pointing at
an ISOLATED test database. This test does not touch candidate data.
"""
import asyncio
import os

import pytest

from app.db import Database


def test_two_database_instances_serialize_same_applicant():
    url = os.environ.get('GATEKEEPER_TEST_POSTGRES_URL')
    if not url:
        pytest.skip('Set GATEKEEPER_TEST_POSTGRES_URL to an isolated PostgreSQL test DB')
    if not url.startswith('postgresql+asyncpg://'):
        pytest.fail('GATEKEEPER_TEST_POSTGRES_URL must use postgresql+asyncpg://')

    async def scenario():
        first_db, second_db = Database(url), Database(url)
        acquired = asyncio.Event()
        release = asyncio.Event()
        second_started = asyncio.Event()
        second_acquired = asyncio.Event()
        try:
            async def first():
                async with first_db.join_request_lock(-100123, 456):
                    acquired.set()
                    await asyncio.wait_for(release.wait(), timeout=10)

            async def second():
                await acquired.wait()
                second_started.set()
                async with second_db.join_request_lock(-100123, 456):
                    second_acquired.set()

            task1 = asyncio.create_task(first())
            task2 = asyncio.create_task(second())
            try:
                await asyncio.wait_for(second_started.wait(), timeout=5)
                # Wait until the other transaction has attempted to acquire lock.
                await asyncio.sleep(0.15)
                assert not second_acquired.is_set(), 'Concurrent DB worker bypassed advisory lock'
            finally:
                release.set()
            await asyncio.wait_for(asyncio.gather(task1, task2), timeout=10)
            assert second_acquired.is_set()
        finally:
            await first_db.close()
            await second_db.close()

    asyncio.run(scenario())
