# Chassis Photo Measurement Feasibility (MVP -> Pro)

## Goal
Estimate chassis geometry from customer-supplied photos (multi-angle) with clear confidence bands:
- wheelbase
- rake/trail proxy
- wheel alignment symmetry
- frame/fork/shock bend indicators

## Bottom-line Feasibility
This is feasible **only with constrained capture workflow** and calibration references.  
Unconstrained phone photos can still provide useful screening signals, but not engineering-grade absolute dimensions.

## Recommended Capability Tiers

### Tier 1: Guided 2D Measurement Assistant (highly feasible now)
- User takes required views with on-screen capture guide:
  - true side view
  - front view
  - rear view
  - optional top-oblique
- Require one or more scale references in-frame:
  - checkerboard/A4 marker
  - known wheel diameter profile
  - optional printed fiducial marker
- Operator-assisted point placement with snap-to-edge helper.

Expected accuracy (typical workshop phone images):
- wheelbase: +/-4 to 8 mm
- axle vertical offset/ride-height deltas: +/-3 to 6 mm
- steering axis proxy: +/-0.4 to 0.8 deg
- trail proxy: +/-4 to 10 mm

Best use:
- setup comparison, before/after checks, crash-screening
- not legal certification, not jig replacement

### Tier 2: Semi-automatic keypoint detection (feasible with model training)
- CV model suggests wheel centers, axle points, fork tube lines, swingarm and frame reference points.
- User confirms/corrects points before computation.
- Strongly improves speed and consistency.

Expected accuracy (after model tuning and good image quality):
- wheelbase: +/-3 to 6 mm
- steering axis proxy: +/-0.3 to 0.6 deg
- trail proxy: +/-3 to 8 mm

### Tier 3: Multi-view reconstruction / 3D (harder, still feasible)
- Multi-image solve (SfM/pose-assisted) with calibration board.
- Better for frame twist and alignment quantification.
- Increased engineering complexity and QA burden.

Expected accuracy (controlled capture protocol):
- wheelbase: +/-2 to 5 mm
- alignment/twist indicators: useful trend detection; hard absolute guarantees without controlled rigs

## Core Technical Requirements
1. **Capture protocol enforcement**
   - camera distance and angle prompts
   - image quality checks (blur, over/under exposure)
   - mandatory calibration object in each frame

2. **Calibration**
   - intrinsic estimate (phone metadata + refinement)
   - planar homography per view
   - scale lock from known marker dimensions

3. **Reference point strategy**
   - wheel center extraction from rim circle fit
   - steering axis from fork tube centerline pair
   - rear/front axle line and chassis centerline proxy
   - uncertainty score per detected feature

4. **Measurement model**
   - compute value + confidence interval, not single hard number
   - fail-safe when confidence below threshold
   - show "insufficient image quality" instead of misleading result

5. **Bend / alignment diagnostics**
   - left/right symmetry deltas from front and rear views
   - wheel plane misalignment proxy
   - fork tube non-parallelism proxy
   - frame straightness warning index (screening only)

## Suggested MVP in This App
1. New "Chassis Photo Check (Beta)" panel.
2. Upload 3 guided photos (side/front/rear).
3. Manual-assisted point pick with CV hint (if available).
4. Compute:
   - wheelbase estimate
   - rake/trail proxy
   - front/rear alignment asymmetry
   - frame-bend risk index (Low / Medium / High)
5. Display confidence bars and quality warnings.
6. Export a short report section with image thumbnails + measured outputs.

## Risks / Constraints
- Perspective distortion and lens distortion dominate error if capture guide ignored.
- Different tire profiles and sag conditions can bias wheel-center and geometry estimates.
- Crash diagnostics need cautious language: "**screening indicator only**".
- Requires clear UX to avoid false confidence.

## Recommended Go/No-Go
Go for Tier 1 now (guided + assisted manual points).  
Parallel-track Tier 2 CV-assisted keypoints after collecting labeled workshop images.

This gives fast customer value with controlled risk while preserving path to deeper automation.
