# IDEA Tracking Efficiency 

This README explains the tracking-efficiency study with the IDEA detector and a muon gun. It explains the concepts in `eff.ipynb`notebook.
---

## 1) Mapping (digi → MC) 

**Goal.** For every *digitized hit* (a “digi”) used by a fitted track, decide which **MC particle** it came from. Once each digi has an MC label, we can count how many track hits belong to each truth particle.

**Rule used — [Majority-of-links](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTrackMCTruth?).** Each digi is linked to one or more **sim-hits**; each sim-hit belongs to an **MC particle**. For that digi, we **count** how many linked sim-hits point to each MC and pick the MC with the **largest count** (ties broken determined by larger MC id).

**Code:**

* `make_digi_to_mc_map(...)` builds a **per-event dictionary** `{digi_index → mc_id}` for each subdetector. It’s called five times:

  ```python
  dch_map, vxb_map, vxd_map, swb_map, swd_map
  ```
* `hit_to_mc_for_event(evt, col_id, idx)` uses those maps to label *each* fitted-track hit by its MC id, based on the hit’s `collectionID`.

---

## 2) Track ↔ truth “best match” and **[Definition 2](https://repository.cern/records/pwrx1-wvn43)** metrics

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

**(d) Compute [Definition 2](https://repository.cern/records/pwrx1-wvn43) metrics for (track, best_mcid).**

* **Purity** = `shared / total_track_hits`
* **Hit-efficiency** = `shared / truth_total`, where `truth_total` is the **SIM-HIT count** for that MC across all 5 subsystems:

  ```python
  truth_hits_by_mc = truth_simhits_all(evt)
  truth_total = truth_hits_by_mc.get(best_mcid, 0)
  ```

**(e) Decide if the track is a *valid match*.**

```python
if (purity >= 0.5) and (hit_eff >= 0.5) and (shared >= nhit_min):
    candidates.append((best_mcid, shared, purity, hit_eff))
```

**(f) One-to-one truth ↔ track.**
If multiple tracks point to the same truth muon, keep the one with the **largest `shared`**:

```python
best_by_truth[mcid] = (shared, pur, hef)
matched_truth_ids = set(best_by_truth.keys())
```

---

## 3) Denominator and numerator (and how they’re built)

 **efficiency vs (p_T)** is measured over **truth muons** in `pt_bins`.

### Denominator: “reconstructable truth muons”

A truth muon contributes to the denominator in its (p_T) bin if:

1. It passes the **truth selection**:

   ```python
   mask_truth = (abs(mc_pdg)==13) & (mc_gen==1) & (from_ip)
   # current 'from_ip' uses Rxy < 50 mm 
   ```

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

> **TL;DR:** **den** = truth muons that can/should be reconstructed; **num** = those for which reconstruction produced a **valid matched track**.

---

## 5) Error bars

**Wald interval ((\approx 68%) CL)** for binomial (k) out of (n):
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
Per (p_T) bin using **binomial Wald** errors with (Z=1).

---


