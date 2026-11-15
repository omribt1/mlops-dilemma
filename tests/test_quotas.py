"""quotas.py: Cloud Quotas parsing (defaults, omitted zeros, Spot ignored) and accelerator -> metric derivation."""
from aiops_dilemma import catalog, quotas

P = "aiplatform.googleapis.com/custom_model_serving_"
INFOS = [
    {"quotaId": "CustomModelServingRTXPRO6000GPUsPerProjectPerRegion", "metric": P + "nvidia_rtx_pro_6000_gpus",
     "dimensions": ["region"], "metricDisplayName": "Custom model serving Nvidia RTX Pro 6000 GPUs",
     "dimensionsInfos": [
         {"dimensions": {"region": "us-central1"}, "details": {"value": "8"}, "applicableLocations": ["us-central1"]},
         {"details": {}, "applicableLocations": ["me-west1", "europe-west1"]}]},
    {"quotaId": "H100", "metric": P + "nvidia_h100_gpus", "dimensions": ["region"],
     "dimensionsInfos": [{"dimensions": {"region": "us-central1"}, "details": {"value": "16"}}]},
    {"quotaId": "A100_80", "metric": P + "nvidia_a100_80gb_gpus", "dimensions": ["region"],
     "dimensionsInfos": [{"dimensions": {"region": "us-central1"}, "details": {"value": "1"}}]},
    {"quotaId": "A100", "metric": P + "nvidia_a100_gpus", "dimensions": ["region"],
     "dimensionsInfos": [{"dimensions": {"region": "us-central1"}, "details": {"value": "14"}}]},
    {"quotaId": "V6E", "metric": P + "tpu_v6e", "dimensions": ["region"],
     "dimensionsInfos": [{"dimensions": {"region": "europe-west4"}, "details": {"value": "8"}}]},
    {"quotaId": "Spot", "metric": P + "preemptible_nvidia_h100_gpus", "dimensions": ["region"],
     "dimensionsInfos": [{"dimensions": {"region": "us-central1"}, "details": {"value": "99"}}]},
    {"quotaId": "Other", "metric": "aiplatform.googleapis.com/online_prediction_requests", "dimensions": ["region"]},
]


def test_parse_defaults_zero_and_spot():
    m = quotas.parse(INFOS)
    assert "preemptible_nvidia_h100_gpus" not in m and len(m) == 5
    r = m["nvidia_rtx_pro_6000_gpus"]["regions"]
    assert r == {"us-central1": 8, "me-west1": 0, "europe-west1": 0}   # omitted value = 0
    assert m["tpu_v6e"]["unit_cores"] and not m["nvidia_h100_gpus"]["unit_cores"]


def test_metric_for_derives_names():
    m = quotas.parse(INFOS)
    assert quotas.metric_for("NVIDIA_RTX_PRO_6000", m) == "nvidia_rtx_pro_6000_gpus"
    assert quotas.metric_for("NVIDIA_H100_80GB", m) == "nvidia_h100_gpus"          # memory suffix dropped
    assert quotas.metric_for("NVIDIA_A100_80GB", m) == "nvidia_a100_80gb_gpus"     # exact name wins
    assert quotas.metric_for("NVIDIA_TESLA_A100", m) == "nvidia_a100_gpus"         # "tesla_" dropped
    assert quotas.metric_for("TPU_V6E", m) == "tpu_v6e"
    assert quotas.metric_for("NVIDIA_L4", m) is None


def test_for_region_and_errors(monkeypatch):
    monkeypatch.setattr(quotas, "fetch", lambda project: {"ok": True, "project": project, "fetched_at": 1,
                                                          "metrics": quotas.parse(INFOS)})
    quotas.clear()
    q = quotas.for_region("proj-a", "me-west1", ["NVIDIA_RTX_PRO_6000", "NVIDIA_L4", "TPU_V6E"])
    assert q["ok"] and q["by_accelerator"]["NVIDIA_RTX_PRO_6000"]["limit"] == 0
    assert q["by_accelerator"]["NVIDIA_L4"]["limit"] is None
    assert q["by_accelerator"]["TPU_V6E"]["unit"] == "cores"
    assert not quotas.for_region("my-gcp-project", "us-central1", ["NVIDIA_L4"])["ok"]

    def boom(project):
        raise OSError("network down")
    monkeypatch.setattr(quotas, "fetch", boom)
    quotas.clear()
    e = quotas.for_region("proj-b", "us-central1", ["NVIDIA_L4"])
    assert not e["ok"] and "network down" in e["error"]
    quotas.clear()


def test_every_gpu_in_chip_table_maps_with_real_metric_names():
    real = {s: {} for s in ["nvidia_l4_gpus", "nvidia_rtx_pro_6000_gpus", "nvidia_a100_gpus", "nvidia_a100_80gb_gpus",
                            "nvidia_h100_gpus", "nvidia_h200_gpus", "nvidia_b200_gpus", "tpu_v5e", "tpu_v6e"]}
    missing = [a for a in catalog.CHIP_SPECS if a != "TPU_V5P" and not quotas.metric_for(a, real)]
    assert not missing
