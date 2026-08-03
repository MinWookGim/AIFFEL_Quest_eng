"""품질 게이트 — 한 편(4컷)이 내보낼 만한가를 판정한다.

흔한 품질 게이트(측정 보고서 + 임계값 표 + 통과 판정)를 쓰되, **우리 문제에 맞춰 두 군데를 바꿨다.**

  1) 양방향 임계값. 보통은 "최소 몇 이상"만 본다. 우리는 컷 간 유사도에 **상한**도 필요하다.
     4컷이 완전히 똑같으면 유사도 1.000 으로 만점인데, 그건 복붙이라 실패다.
  2) 무작위·천장을 같이 들고 다닌다. 보통은 "정확도 0.95 이상" 같은 절대선을 쓴다.
     우리 지표는 천장이 1.0 이 아니다 — **진짜 그림도 만점을 못 받는다.** 그래서 절대선이 무의미하다.

임계값 근거 (전부 실측. `verify/consistency_baseline_v2.py`, 진짜 그림 4장짜리 편 2000개)
------------------------------------------------------------------------------------
                          무작위    CLIP     Gram VGG19
  컷 단위 라벨 정확도       0.111    0.296    0.440
  편 style_precision 중앙   0.111    0.250    0.500     <- 이게 천장이다
  4컷 전부일치             ~0.000   0.021    0.065

  주의: 생성 평가에는 **Gram 을 쓴다.** CLIP 은 천장이 0.25 라 잴 수 있는 폭이 너무 좁다.

  Gram 기준 style_precision 분포:  10%=0.000  25%=0.250  중앙=0.500  75%=0.500
    - 0.25(4컷 중 1컷) 이상: 진짜 그림 84.4% 통과 / 무작위도 37.6% 통과  -> 너무 헐겁다
    - 0.50(4컷 중 2컷) 이상: 진짜 그림 53.8% 통과 / 무작위는  6.4% 통과  -> **여기가 통과선**
      (무작위 확률은 이항분포 B(4, 1/9) 로 계산)

  컷 간 절대 코사인은 판별에 실패했다 (같은 그림체 0.669 vs 다른 그림체 0.656).
  그래서 **판정에는 안 쓰고, 복붙 탐지 상한으로만 남긴다** (같은 그림 4장 = 1.000).
"""

# 한 편을 재고 나면 이 모양의 보고서가 나온다
EXAMPLE_REPORT = {
    "case_id": "G-m3-01",
    "requested_style": "m3",
    "panels_expected": 4,
    "panels_produced": 4,      # 실제로 나온 컷 수      -> panel recall
    "style_precision": 0.50,   # 4컷 중 요청 라벨로 판정된 비율 (Gram 기준)
    "edition_cosine": 0.71,    # 4컷 서로의 코사인 평균 (6쌍)  -> 복붙 탐지용
    "latency_sec": 84.3,       # 한 편 만드는 데 걸린 시간
    "cost_usd": 0.182,
}

QUALITY_THRESHOLDS = {
    # ── 타협 없는 것 ───────────────────────────────
    "panel_recall_min": 1.00,       # 4컷은 다 나와야 한다. 한 컷 빠지면 만화가 아니다

    # ── 그림체 (Gram 기준. 근거는 파일 맨 위) ──────
    "style_precision_min": 0.50,    # 진짜 그림 중앙값. 무작위(0.111)가 통과할 확률 6.4%
    # 목표는 **정규화 점수**로 잡는다. 절대값으로 잡으면 천장(0.50)보다 높은 목표를 세우는
    # 자기모순이 생긴다 (실제로 처음에 0.75로 잡았다가 "정규화 1.00인데 진품급 아님"이 떴다).
    "style_normalized_target": 1.0,  # 1.0 = 진짜 그림 중앙값만큼 했다

    # ── 복붙 방지 (상한) ─────────────────────────
    "edition_cosine_max": 0.95,     # 같은 그림 4장이면 1.000. 정상 범위는 0.584~0.773

    # ── 팔 수 있는가 ──────────────────────────────
    "latency_sec_max": 120.0,       # 실측 장당 19.2초 -> 4컷 76.8초. 여유 두고 2분
    "cost_usd_max": 0.30,           # 실측 4컷 $0.182
}

CHANCE_STYLE_PRECISION = 0.111      # 9클래스 무작위
CEILING_STYLE_PRECISION = 0.50      # Gram 기준 진짜 그림 중앙값


def normalized_style_score(style_precision):
    """천장 대비로 정규화한다.

    절대값 0.5 는 "절반밖에 못 맞혔다"로 읽히지만, 진짜 그림도 0.5 다.
    그래서 무작위를 0, 천장을 1 로 놓고 다시 잰다. 1.0 이면 진품만큼 한 것이다.
    """
    lo, hi = CHANCE_STYLE_PRECISION, CEILING_STYLE_PRECISION
    return (style_precision - lo) / (hi - lo)


def check_quality_gate(report, thresholds=QUALITY_THRESHOLDS):
    """내보낼 수 있나 판정한다. 막는 이유(failed)와 눈여겨볼 것(warnings)을 나눠 돌려준다."""
    failed, warnings = [], []

    panel_recall = report["panels_produced"] / report["panels_expected"]
    if panel_recall < thresholds["panel_recall_min"]:
        failed.append(
            f"컷이 빠졌다 — {report['panels_produced']}/{report['panels_expected']}컷 "
            f"(recall {panel_recall:.2f}). 고칠 곳: 출력 스키마 검증")

    sp = report["style_precision"]
    if sp < thresholds["style_precision_min"]:
        failed.append(
            f"그림체가 안 잡혔다 — style precision {sp:.2f} < {thresholds['style_precision_min']:.2f} "
            f"(무작위 {CHANCE_STYLE_PRECISION:.3f} / 천장 {CEILING_STYLE_PRECISION:.2f}). "
            f"고칠 곳: 레퍼런스 선택")
    elif normalized_style_score(sp) < thresholds["style_normalized_target"]:
        warnings.append(
            f"통과선은 넘었지만 진품 중앙값에는 못 미친다 "
            f"(정규화 {normalized_style_score(sp):.2f} < {thresholds['style_normalized_target']:.2f})")

    ec = report.get("edition_cosine")
    if ec is not None and ec > thresholds["edition_cosine_max"]:
        failed.append(
            f"4컷이 사실상 같은 그림이다 — 컷 간 코사인 {ec:.3f} > {thresholds['edition_cosine_max']:.2f}. "
            f"일관성이 높은 게 아니라 복붙이다")

    lat = report.get("latency_sec")
    if lat is not None and lat > thresholds["latency_sec_max"]:
        warnings.append(f"느리다 — {lat:.0f}초 > {thresholds['latency_sec_max']:.0f}초. 팔기 어렵다")

    cost = report.get("cost_usd")
    if cost is not None and cost > thresholds["cost_usd_max"]:
        warnings.append(f"비싸다 — ${cost:.3f} > ${thresholds['cost_usd_max']:.2f}")

    return {
        "case_id": report.get("case_id"),
        "can_deploy": len(failed) == 0,
        "failed_reasons": failed,
        "warnings": warnings,
        "panel_recall": round(panel_recall, 3),
        "style_precision": sp,
        "style_normalized": round(normalized_style_score(sp), 3),
    }


def summarize(results):
    """여러 편을 한 번에 봤을 때 — 평균이 숨기는 걸 드러낸다.

    strict accuracy = 통째로 통과한 편의 비율. 부분점수 평균이 좋아도 이게 0 일 수 있다.
    """
    n = len(results)
    ok = sum(r["can_deploy"] for r in results)
    reasons = {}
    for r in results:
        for f in r["failed_reasons"]:
            key = f.split("—")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
    return {
        "편수": n,
        "strict_accuracy": round(ok / n, 3) if n else 0.0,
        "통과": ok,
        "탈락": n - ok,
        "평균_style_normalized": round(sum(r["style_normalized"] for r in results) / n, 3) if n else 0.0,
        "탈락사유": dict(sorted(reasons.items(), key=lambda x: -x[1])),
    }


if __name__ == "__main__":
    import json

    # 손으로 만든 예시 4편으로 게이트가 제대로 막고 통과시키는지 본다
    cases = [
        dict(EXAMPLE_REPORT),                                                    # 정상
        {**EXAMPLE_REPORT, "case_id": "G-m3-02", "panels_produced": 3},          # 컷 누락
        {**EXAMPLE_REPORT, "case_id": "G-m5-01", "style_precision": 0.25},       # 그림체 실패
        {**EXAMPLE_REPORT, "case_id": "G-m2-01", "edition_cosine": 0.985},       # 복붙
    ]
    results = [check_quality_gate(c) for c in cases]
    for r in results:
        mark = "통과" if r["can_deploy"] else "탈락"
        print(f"[{mark}] {r['case_id']}  style정규화 {r['style_normalized']:+.2f}")
        for f in r["failed_reasons"]:
            print(f"        X {f}")
        for w in r["warnings"]:
            print(f"        ! {w}")
    print()
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2))
