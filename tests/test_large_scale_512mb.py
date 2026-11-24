"""Large-scale regression test: ~50k records within a 512 MiB address space.

Simulates the registrar's case: two CSVs totalling ~50k records with at
most two records per minute.  Both the pure verifier and the full web flow
(case creation, supplement re-judgement, conflict rejection and the case
page) must complete with the process address space capped at 512 MiB.

Run directly (it temporarily lowers RLIMIT_AS while each check executes):

    python tests/test_large_scale_512mb.py
"""

import datetime as dt
import io
import os
import resource
import sys
import tempfile
import time

sys.path.insert(0, "/workspace")

from verifier import (  # noqa: E402
    TABLE_QC,
    TABLE_WAREHOUSE,
    analyze,
    build_graph,
    parse_csv,
)

try:
    from app import create_app
except ImportError:  # Flask is optional for the verifier-only check.
    create_app = None

LIMIT = 512 * 1024 * 1024
N_PAIRS = 25000  # 25k warehouse + 25k qc = 50k records, <= 2 per minute


def make_csvs(n_pairs=N_PAIRS):
    base = dt.datetime(2026, 1, 1, 0, 0)
    warehouse = io.StringIO()
    warehouse.write("记录号,分钟时间,动作\n")
    qc = io.StringIO()
    qc.write("记录号,分钟时间,动作\n")
    for i in range(n_pairs):
        minute = (base + dt.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M")
        warehouse.write(
            f"W-{i:05d},{minute},{'交运' if i == n_pairs - 1 else '换签'}\n"
        )
        qc.write(f"Q-{i:05d},{minute},{'验封通过' if i % 3 else '验封失败'}\n")
    return warehouse.getvalue().encode(), qc.getvalue().encode()


class limit_address_space:
    def __enter__(self):
        self.saved = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (LIMIT, self.saved[1]))
        return self

    def __exit__(self, *exc):
        resource.setrlimit(resource.RLIMIT_AS, self.saved)
        return False


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_CSV_CACHE = None


def get_csvs():
    global _CSV_CACHE
    if _CSV_CACHE is None:
        _CSV_CACHE = make_csvs()
    return _CSV_CACHE


def check_order_valid(records, supplements, uid_order):
    n = len(records)
    assert uid_order is not None and len(uid_order) == n
    assert len(set(uid_order)) == n
    pos = {uid: i for i, uid in enumerate(uid_order)}
    adj, _ = build_graph(records, supplements)
    for u, vs in enumerate(adj):
        for v in vs:
            assert pos[records[u]["uid"]] < pos[records[v]["uid"]]


def test_verifier_within_limit():
    warehouse_csv, qc_csv = get_csvs()
    with limit_address_space():
        t0 = time.perf_counter()
        records = parse_csv(warehouse_csv, TABLE_WAREHOUSE) + parse_csv(
            qc_csv, TABLE_QC
        )
        assert len(records) == 2 * N_PAIRS
        result = analyze(records, [])
        elapsed = time.perf_counter() - t0
        assert result["record_count"] == 2 * N_PAIRS
        assert result["verdict"] in {
            "ambiguous",
            "always_compliant",
            "always_noncompliant",
        }
        for key in ("compliant_order", "noncompliant_order"):
            if result[key] is not None:
                check_order_valid(records, [], result[key])
        print(
            f"verifier: {len(records)} records analysed in {elapsed:.1f}s, "
            f"verdict={result['verdict']}, peak={peak_mib():.0f} MiB"
        )


def test_web_flow_within_limit():
    if create_app is None:
        print("web flow: skipped (Flask not installed)")
        return
    warehouse_csv, qc_csv = get_csvs()
    fd, db_path = tempfile.mkstemp(prefix="boxcheck-50k-", suffix=".db")
    os.close(fd)
    os.unlink(db_path)
    try:
        with limit_address_space():
            app = create_app({"TESTING": True, "DATABASE_PATH": db_path})
            client = app.test_client()

            t0 = time.perf_counter()
            resp = client.post(
                "/api/cases",
                data={
                    "box_code": "BIG-50K",
                    "warehouse_file": (io.BytesIO(warehouse_csv), "w.csv"),
                    "qc_file": (io.BytesIO(qc_csv), "q.csv"),
                },
                content_type="multipart/form-data",
            )
            assert resp.status_code == 201, resp.get_data(as_text=True)[:300]
            detail = resp.get_json()
            case_id = detail["case"]["id"]
            conclusion = detail["conclusion"]
            assert conclusion["record_count"] == 2 * N_PAIRS
            for key in ("compliant_order", "noncompliant_order"):
                if conclusion[key] is not None:
                    assert len(conclusion[key]) == 2 * N_PAIRS
            print(f"web create: 201 in {time.perf_counter() - t0:.1f}s")
            del resp, detail, conclusion

            t0 = time.perf_counter()
            resp = client.post(
                f"/api/cases/{case_id}/supplements",
                json={"before_uid": "W::W-00005", "after_uid": "Q::Q-00005"},
            )
            assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
            print(f"web supplement: 200 in {time.perf_counter() - t0:.1f}s")
            del resp

            resp = client.post(
                f"/api/cases/{case_id}/supplements",
                json={"before_uid": "Q::Q-00005", "after_uid": "W::W-00005"},
            )
            assert resp.status_code == 409
            body = resp.get_json()
            assert len(body["state"]["supplements"]) == 1
            print("web conflict: 409, original supplement and verdict kept")
            del resp, body

            t0 = time.perf_counter()
            resp = client.get(f"/cases/{case_id}")
            assert resp.status_code == 200
            total = 0
            for piece in resp.response:  # streamed like a real client read
                total += len(piece)
            assert total > 0
            print(
                f"web case page: 200, {total / 1e6:.1f} MB html "
                f"in {time.perf_counter() - t0:.1f}s"
            )
            del resp

            t0 = time.perf_counter()
            resp = client.post(
                "/cases",
                data={
                    "box_code": "BIG-50K-HTML",
                    "warehouse_file": (io.BytesIO(warehouse_csv), "w.csv"),
                    "qc_file": (io.BytesIO(qc_csv), "q.csv"),
                },
                content_type="multipart/form-data",
            )
            assert resp.status_code == 303
            location = resp.headers["Location"]
            del resp
            resp = client.get(location)
            assert resp.status_code == 200
            for _piece in resp.response:
                pass
            del resp
            print(f"web form create + page: ok in {time.perf_counter() - t0:.1f}s")

            print(f"web flow peak: {peak_mib():.0f} MiB (limit 512 MiB)")
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def main():
    test_verifier_within_limit()
    test_web_flow_within_limit()
    print(f"OK: 50k-record case fits in 512 MiB (peak {peak_mib():.0f} MiB)")


if __name__ == "__main__":
    main()
