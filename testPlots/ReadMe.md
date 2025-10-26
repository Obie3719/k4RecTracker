# IDEA Tracking Efficiency — Method & Implementation Guide

This README explains the **what/why/how** of the tracking-efficiency study you’re running with the IDEA detector and a muon gun. It ties each concept to **exact places in your notebook**, so you can point to the code confidently.

---

## 1) What is being mapped (digi → MC) and how

**Goal.** For every *digitized hit* (a “digi”) used by a fitted track, decide which **MC particle** it came from. Once each digi has an MC label, we can count how many track hits belong to each truth particle.

**Rule used — Majority-of-links.** Each digi is linked to one or more **sim-hits**; each sim-hit belongs to an **MC particle**. For that digi, we **count** how many linked sim-hits point to each MC and pick the MC with the **largest count** (ties broken deterministically by larger MC id).

**Where in code.**

* `make_digi_to_mc_map_majority(...)` builds a **per-event dictionary** `{digi_index → mc_id}` for each subdetector. It’s called five times:

  ```python
  dch_map, vxb_map, vxd_map, swb_map, swd_map
  ```
* `hit_to_mc_for_event(evt, col_id, idx)` uses those maps to label *each* fitted-track hit by its MC id, based on the hit’s `collectionID` (which subdetector map to query).

**Why this rule.** It’s simple, weight-free, and robust for clean muon-gun samples. (If reliable per-link weights exist, you can swap to a **max-weight** rule with the same machinery.)

---

## 2) Track ↔ truth “best match” and **Definition 2** metrics

For each **reconstructed track**:

**(a) Countable hits on the track.**
Only hits from the 5 IDEA subsystems are counted:

```python
COUNTABLE_CIDS = set(CID.values())   # all five
```

Inside the track loop, `total_track_hits` increments only for these.

**(b) Assign those hits to truth.**
For each counted hit:

```python
contrib[mcid] += 1
```

So `contrib[mcid]` = **# of track hits** that belong to MC particle `mcid`.

**(c) Choose the best truth particle for the track.**

```python
best_mcid, shared = contrib.most_common(1)[0]
```

* **`shared`** = number of track hits that match **best_mcid**.

**(d) Compute Definition 2 metrics for (track, best_mcid).**

* **Purity** = `shared / total_track_hits`
* **Hit-efficiency** = `shared / truth_total`, where `truth_total` is the **SIM-HIT count** for that MC across all 5 subsystems:

  ```python
  truth_hits_by_mc = truth_simhits_all(evt)
  truth_total = truth_hits_by_mc.get(best_mcid, 0)
  ```

**(e) Decide if the track is a *valid match*.**
Apply **Definition 2** + a practical shared-hit guard:

```python
if (purity >= 0.5) and (hit_eff >= 0.5) and (shared >= nhit_min):
    candidates.append((best_mcid, shared, purity, hit_eff))
```

This enforces **clean** (purity), **complete** (hit-eff), and **non-trivial** (shared) matches.

**(f) One-to-one truth ↔ track.**
If multiple tracks point to the same truth muon, keep the one with the **largest `shared`**:

```python
best_by_truth[mcid] = (shared, pur, hef)
matched_truth_ids = set(best_by_truth.keys())
```

---

## 3) What are the **denominator** and **numerator** (and how they’re built)

You measure **efficiency vs (p_T)** over **truth muons** in `pt_bins`.

### Denominator: “reconstructable truth muons”

A truth muon contributes to the denominator in its (p_T) bin if:

1. It passes the **truth selection**:

   ```python
   mask_truth = (abs(mc_pdg)==13) & (mc_gen==1) & (from_ip)
   # current 'from_ip' uses Rxy < 0.1 mm in your notebook
   ```

   *(You can tighten later to θ cuts and Rxy < 50 mm if desired.)*

2. It has **enough truth activity** in the detector using **SIM-HITS** across all 5 subdetectors:

   ```python
   truth_hits_by_mc = truth_simhits_all(evt)  # Counter {mcid: n_simhits}
   if truth_hits_by_mc.get(tid, 0) >= nhit_min:
       recon_ids.append(tid); recon_pts.append(pt)
   ```

3. Then you bin those `recon_pts` and fill:

   ```python
   den[b] += 1
   ```

### Numerator: “reconstructed truth muons”

Among the **same reconstructable** muons in each (p_T) bin, a muon contributes to the numerator if it is in `matched_truth_ids` **after one-to-one matching**:

```python
if tid in matched_truth_ids:
    num[b] += 1
```

> **TL;DR:** **den** = truth muons that can/should be reconstructed; **num** = those for which reconstruction produced a **valid matched track** (Def. 2).

---

## 4) Where each piece lives in the notebook

* **Subdetector IDs & counted hits:** `CID{...}`, `COUNTABLE_CIDS` (top).
* **Truth arrays:** momenta, PDG, generator status, vertices (first `t.arrays([...])`).
* **Fitted-track hit references:** `ft_beg`, `ft_end`, `ft_idx`, `ft_cid`.
* **Sim-hit → MC for each subdetector:** the `_..._particle.index` arrays loaded in `t.arrays([...])`.
* **Digi→MC mapping (majority):** `make_digi_to_mc_map_majority(...)` + the five `*_map`.
* **Track-hit → MC assignment:** `hit_to_mc_for_event(...)`.
* **Truth SIM-HIT totals:** `truth_simhits_all(evt)`.
* **Main event loop:** builds `recon_ids/recon_pts` (denominator), assembles `candidates`, applies Def. 2, enforces one-to-one, then fills `num/den`.
* **Binning:** `np.digitize(recon_pts, pt_bins) - 1`.
* **Plots:** two `plt.errorbar(...)` blocks at the end.

---

## 5) Error bars

Two plots are produced:

1. **Points only** (no y-errors):
   `eff_points = num / den` per bin.

2. **Wald interval ((\approx 68%) CL)** for binomial (k) out of (n):
   [
   \hat p = \frac{k}{n}, \qquad
   \sigma_{\text{Wald}} = \sqrt{\frac{\hat p(1-\hat p)}{n}}, \qquad
   \text{error} = \pm Z,\sigma_{\text{Wald}} \ \ (\text{with } Z=1)
   ]

   ```python
   p_hat = k / n
   sigma = math.sqrt(max(p_hat*(1.0 - p_hat)/n, 0.0))
   err   = Z * sigma
   plt.errorbar(..., yerr=err, ...)
   ```

   For small (n) or (p) near 0/1, consider **Wilson** intervals for better coverage.

---

## 6) Why this setup is sound for IDEA + muon gun

* **Consistent “hit” definitions:** hit-eff denominator uses **SIM-HITS** (truth), while purity and the track-side numerator rely on the same **digi→MC** mapping.
* **All subdetectors** are included in both the track-hit counting (purity denominator) and truth sim-hit totals (hit-eff denominator).
* **Definition 2** is balanced (clean *and* complete).
* **One-to-one** matching avoids double-counting.
* With a clean one-muon gun inside acceptance, efficiencies near **100%** at moderate/high (p_T) are expected.

---

## Quick FAQ

**What exactly is “mapping”?**
Turning raw track hits into *truth-labeled* hits via digi→MC association (majority-of-links). It lets us compute overlap (shared hits) between each track and each truth particle.

**What are “purity” and “hit-efficiency”?**
Purity = fraction of a track’s counted hits that come from a **single** truth muon.
Hit-efficiency = fraction of that muon’s **truth sim-hits** that are picked up by the matched track.

**What is the denominator (“reconstructable”)?**
Truth muons with **≥ `nhit_min`** truth sim-hits across the five subsystems (after truth selections).

**What is the numerator (“reconstructed”)?**
Those denominator muons that have **one** matched track with **purity ≥ 50%** and **hit-eff ≥ 50%**, and **shared ≥ `nhit_min`**.

**Where are `num` and `den` filled?**
Inside the event loop, after one-to-one matching:

```python
den[b] += 1
if tid in matched_truth_ids:
    num[b] += 1
```

**How are errors computed?**
Per (p_T) bin using **binomial Wald** errors with (Z=1) (optionally swap to **Wilson**).

---

### Optional: tighter acceptance (to mirror the note)

If you want the exact analysis acceptance used in many studies, add:

```python
mc_p  = np.sqrt(mc_px**2 + mc_py**2 + mc_pz**2)
mc_th = np.arccos(np.clip(mc_pz / np.maximum(mc_p, 1e-12), -1.0, 1.0))
theta_mask  = (mc_th > np.deg2rad(10)) & (mc_th < np.deg2rad(170))
from_ip50mm = (np.sqrt(mc_vx**2 + mc_vy**2) < 50.0)
truth_preselect = (abs(mc_pdg)==13) & (mc_gen==1) & theta_mask & from_ip50mm
```

---

*If you’d like, add this README alongside your notebook and I can annotate your code with short comments that mirror these sections—so anyone opening the repo can jump between README and source instantly.*
