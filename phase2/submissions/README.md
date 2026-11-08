# Phase 2 submissions

Every Phase 2 (L-NER) archive built for Codabench, moved here from Downloads and kept under its build tag, so each name matches the tag in EXPERIMENTS.md and on Drive (`phase2/submissions/`). Each zip contains a single `prediction.json`.

Scores are filled in only where EXPERIMENTS.md or a recorded Codabench row gives them. A dash means the file was built but its score was not recorded here. See EXPERIMENTS.md for the experiment behind each file.

**Graded submission: `submission_r3_rescue.zip`, 0.6515.**

| built | file | what it is | test |
|---|---|---|---|
| 10-07 19:54 | `submission_M1_SC_bi.zip` | M1_SC, first NER fine-tune | – |
| 10-08 07:24 | `submission_M1_SC_hdr.zip` | M1_SC with header handling | – |
| 10-08 13:25 | `submission_ens4.zip` | 4-member ensemble, model only | 0.4435 |
| 10-08 15:34 | `submission_ens6.zip` | 6-member ensemble, model only | 0.4522 |
| 10-08 15:34 | `submission_ens6_m4.zip` | 6 members, M4 tripled, model only | 0.4608 |
| 10-08 16:09 | `submission_ens6_m4_scg.zip` | + cross-doc gazetteer | – |
| 10-08 16:09 | `submission_ens6_m4_snap.zip` | + snap | – |
| 10-08 16:13 | `submission_ens6_m4_scwg.zip` | + word-snap and gazetteer | – |
| 10-08 16:13 | `submission_ens6_m4_scwp2g.zip` | + occurrence propagation | – |
| 10-08 16:44 | `submission_ens6_m4_scwg_rp.zip` | + rules, snap only | – |
| 10-08 16:44 | `submission_ens6_m4_scwg_rpadd.zip` | + rules, snap and add | – |
| 10-08 16:48 | `submission_ens6_m4_scwg_rXACRWpadd.zip` | + rules, replace variant | – |
| 10-08 17:04 | `submission_ens6_m4_scwg_rXCDWpadd.zip` | + rules, replace variant | – |
| 10-08 17:04 | `submission_ens6_m4_scwg_rXDpadd.zip` | + rules, replace DATE | 0.5337 |
| 10-08 17:35 | `submission_distil.zip` | knowledge-distillation run (mandatory) | – |
| 10-08 20:53 | `submission_ens6_m4_scwg_rMp.zip` | rules variant | – |
| 10-08 20:53 | `submission_ens6_m4b.zip` | ens6_m4 variant | – |
| 10-10 08:11 | `submission_ens6_m4_scwg_rMp_v2.zip` | later rebuild of rMp; different predictions | – |
| 10-10 16:23 | `submission_wsens6.zip` | weak-supervision ensemble, model only | – |
| 10-10 16:25 | `submission_wsens6_scwg_rpadd.zip` | wsens6, full post-processing | 0.5577 |
| 10-10 16:26 | `submission_wsens6_scwg.zip` | wsens6, word-snap and gazetteer | – |
| 10-11 02:05 | `submission_ext6.zip` | + InLegalNER, model only | 0.4873 |
| 10-11 08:57 | `submission_vote3.zip` | exact 2-of-3 span vote | 0.5763 |
| 10-11 08:58 | `submission_ext6_scwg_rpadd.zip` | ext6, full post-processing | 0.5527 |
| 10-11 09:03 | `submission_vote3all.zip` | exact 3-of-3 span vote | 0.5626 |
| 10-11 09:14 | `submission_vote3o.zip` | 2-of-3 by overlap, base extent | 0.5631 |
| 10-11 09:24 | `submission_stack3b.zip` | stacker over 3 systems | 0.5847 |
| 10-11 10:15 | `submission_stack9.zip` | stacker over 3 systems x 3 stages | 0.6421 |
| 10-11 10:33 | `submission_stack9r.zip` | stack9 variant | – |
| 10-11 10:42 | `submission_stack9rs.zip` | stack9 + context + label swap | 0.6122 |
| 10-11 10:53 | `submission_stack9_lo.zip` | stack9, lower threshold | – |
| 10-11 10:53 | `submission_stack9_hi.zip` | stack9, higher threshold | – |
| 10-11 10:57 | `submission_stack9b.zip` | stack9 variant | – |
| 10-11 12:02 | `submission_stack9p2_r128.zip` | + extent perturbation, 128 spans/doc | 0.6284 |
| 10-11 12:21 | `submission_ext6st_avgmass+0p5_scwg_rpadd.zip` | self-trained ext6st, single system | – |
| 10-11 12:29 | `submission_s9x_r136.zip` | stack9 at 136 spans/doc | 0.6436 |
| 10-11 12:29 | `submission_s9x_r140.zip` | stack9 at 140 spans/doc | 0.6384 |
| 10-11 12:36 | `submission_s9bag_r132.zip` | + bagging, 132 spans/doc | 0.6306 |
| 10-11 12:36 | `submission_s9rich_r132.zip` | + context features, 132 spans/doc | 0.6316 |
| 10-11 14:48 | `submission_stack27_r136.zip` | 4th model, six decoders x 3 stages | 0.6318 |
| 10-11 14:48 | `submission_stack27st_r136.zip` | stack27 with the self-trained model | – |
| 10-11 14:49 | `submission_stack12_r136.zip` | 4th model `ext6e`, mass decoder | 0.6322 |
| 10-11 14:59 | `submission_s9trim.zip` | precision trim of s9x_r136 | 0.6444 |
| 10-11 15:09 | `submission_t3.zip` | trim + near-unanimous adds | 0.6440 |
| 10-11 15:13 | `submission_s9trim152.zip` | trim on a 152-spans/doc base | 0.6376 |
| 10-11 19:19 | `submission_r2_strim.zip` | trim by the self-trained student | 0.6372 |
| 10-11 19:19 | `submission_r1_rescue.zip` | s9trim + student rescue, round 1 | 0.6488 |
| 10-11 19:39 | `submission_v1_wide.zip` | rescue from a wider pool | 0.6463 |
| 10-11 21:20 | `submission_r3_rescue.zip` | rescue, round 3 (teacher r1_rescue) | **0.6515, graded** |
| 10-11 21:48 | `submission_r3s2_rescue.zip` | r3_rescue + round-2 student backing | 0.6515 |
| 10-11 23:12 | `submission_r4_rescue.zip` | rescue, round 4 (teacher r3_rescue) | 0.6514 |

Two exact duplicates in Downloads (`submission_M1_SC_bi (1).zip` and `submission_ens6_m4_scwg_rXACRWpadd (1).zip`, both with an identical `prediction.json`) were sent to the Recycle Bin.
