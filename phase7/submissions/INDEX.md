# Phase 7 submissions -- index

Files are named `<test score>__<system tag>__<run>.zip`. Scores are the Codabench test
Micro-F1@7 from the leaderboard; `noscore` = uploaded but not on the pasted pages.
Citator runs: run1 = rules A/B/D/L (7 Nov 22:14), run2 = + rule N (22:28),
run3 = + High Court scan (8 Nov 08:57), run4 = + mask-free SC rescan (10:42);
sweep_* = the 8 Nov threshold / tier / rule-M / base / group experiments (all <= 0.5157).
Variant letters: a/b = citator rules A/B, d = date demotion, l = corpus lookup (kind=lookup),
v = rules vote on order, t = loose name band tiered, g = citator groups.

**Final pushed submission: `00_FINAL_best_0.5157/0.5157__cit_abdl_ot2_zst3_p__run4.zip`**
(Codabench ID 1041551). Required LoRMA + KD runs are in `04_required_lorma_kd/`.
Not on this laptop (uploaded from the phone via Drive): ot2_zst3_p 0.4536, ot2_zst3_pdq 0.4513,
ot2_zlex2_pdq 0.4412; their zips are in Drive under cs781_capstone/phase7/submissions/_upload.
Removed as exact duplicates (same predictions as `0.5151__cit_abdl_ot2_zst3_p__sweep_t3.zip`):
the `cit_abdl` zips of the `_m` and `_g6` sweeps (both scored 0.5151).

| folder | file | test F1 |
|---|---|---|
| 00_FINAL_best_0.5157 | 0.5157__cit_abdl_ot2_zst3_p__run4.zip | 0.5157 |
| 01_single_systems | 0.4105__citep__07nov.zip | 0.4105 |
| 01_single_systems | 0.3745__tfidf_s015__05nov.zip | 0.3745 |
| 01_single_systems | 0.3633__bm25_x__05nov.zip | 0.3633 |
| 01_single_systems | 0.3337__tfidf_bin__05nov.zip | 0.3337 |
| 01_single_systems | 0.3095__dense_m2_c8_max__05nov.zip | 0.3095 |
| 01_single_systems | 0.2865__dense_m2_c8__05nov.zip | 0.2865 |
| 01_single_systems | 0.2646__bm25_lo__05nov.zip | 0.2646 |
| 01_single_systems | 0.2593__dense_m4__05nov.zip | 0.2593 |
| 01_single_systems | 0.2558__dense_m3__05nov.zip | 0.2558 |
| 01_single_systems | 0.1961__dense_m1_bidir__05nov.zip | 0.1961 |
| 01_single_systems | noscore__bm25__05nov.zip |  |
| 01_single_systems | noscore__bm25_hi__05nov.zip |  |
| 01_single_systems | noscore__dense_m1__05nov.zip |  |
| 01_single_systems | noscore__dense_m2__05nov.zip |  |
| 01_single_systems | noscore__dense_m2_max__05nov.zip |  |
| 01_single_systems | noscore__tfidf__05nov.zip |  |
| 01_single_systems | noscore__tfidf_s03__05nov.zip |  |
| 01_single_systems | noscore__tfidf_sqrt__05nov.zip |  |
| 02_fusion_and_stacks | 0.4241__stack_new__07nov.zip | 0.4241 |
| 02_fusion_and_stacks | 0.4229__stack_ce__07nov.zip | 0.4229 |
| 02_fusion_and_stacks | 0.4223__stack_ce_lex__07nov.zip | 0.4223 |
| 02_fusion_and_stacks | 0.4217__stack_lex2__07nov.zip | 0.4217 |
| 02_fusion_and_stacks | 0.4070__stack__05nov.zip | 0.4070 |
| 02_fusion_and_stacks | 0.4058__stack_d100__05nov.zip | 0.4058 |
| 02_fusion_and_stacks | 0.4046__stack_slow__05nov.zip | 0.4046 |
| 02_fusion_and_stacks | 0.4022__stack_c8__05nov.zip | 0.4022 |
| 02_fusion_and_stacks | 0.4022__stack_lex__05nov.zip | 0.4022 |
| 02_fusion_and_stacks | 0.3987__stack_deep__05nov.zip | 0.3987 |
| 02_fusion_and_stacks | 0.3963__stack_d20__05nov.zip | 0.3963 |
| 02_fusion_and_stacks | 0.3739__fuse_top__05nov.zip | 0.3739 |
| 02_fusion_and_stacks | 0.3278__fuse_all__05nov.zip | 0.3278 |
| 02_fusion_and_stacks | 0.3231__stack_model__05nov.zip | 0.3231 |
| 02_fusion_and_stacks | noscore__fuse__05nov.zip |  |
| 02_fusion_and_stacks | noscore__fuse2__05nov.zip |  |
| 03_transductive_ot | 0.4477__ot_zmix_tr__07nov.zip | 0.4477 |
| 03_transductive_ot | 0.4477__ot_zmix_trdv__07nov.zip | 0.4477 |
| 04_required_lorma_kd | 0.3325__ft_lorma_m2__07nov.zip | 0.3325 |
| 04_required_lorma_kd | 0.3154__kd_half__07nov.zip | 0.3154 |
| 05_citator_graph | 0.5157__cit_abdlv_ot2_zst3_p__run4.zip | 0.5157 |
| 05_citator_graph | 0.5151__cit_abdl_ot2_zst3_p__sweep_n100.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdl_ot2_zst3_p__sweep_n90.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdl_ot2_zst3_p__sweep_t3.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdlt_ot2_zst3_p__sweep_t3.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdlv_ot2_zst3_p__sweep_m.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdlv_ot2_zst3_p__sweep_n100.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdlv_ot2_zst3_p__sweep_n90.zip | 0.5151 |
| 05_citator_graph | 0.5151__cit_abdlvt_ot2_zst3_p__sweep_t3.zip | 0.5151 |
| 05_citator_graph | 0.5139__cit_abdlg_ot2_zst3_p__sweep_g6.zip | 0.5139 |
| 05_citator_graph | 0.5121__cit_abdl_ot2_zst5_p__sweep_base.zip | 0.5121 |
| 05_citator_graph | 0.5121__cit_abl_ot2_zst3_p__run4.zip | 0.5121 |
| 05_citator_graph | 0.5121__cit_ablv_ot2_zst3_p__run4.zip | 0.5121 |
| 05_citator_graph | 0.5115__cit_abl_ot2_zst3_p__sweep_m.zip | 0.5115 |
| 05_citator_graph | 0.5092__cit_abdl_ot2_zst3_p__run3.zip | 0.5092 |
| 05_citator_graph | 0.5080__cit_abdl_ot_zmix_trdv__sweep_base.zip | 0.5080 |
| 05_citator_graph | 0.5062__cit_abl_ot2_zst3_p__run3.zip | 0.5062 |
| 05_citator_graph | 0.5044__cit_abdl_ot2_zst3_p__run2.zip | 0.5044 |
| 05_citator_graph | 0.5015__cit_abl_ot2_zst3_p__run2.zip | 0.5015 |
| 05_citator_graph | 0.5009__cit_abdl_stack_new__sweep_base.zip | 0.5009 |
| 05_citator_graph | 0.5009__cit_al_ot2_zst3_p__run3.zip | 0.5009 |
| 05_citator_graph | 0.4997__cit_abdl_ot2_zst3_p__run1.zip | 0.4997 |
| 05_citator_graph | 0.4979__cit_abd_ot2_zst3_p__run2.zip | 0.4979 |
| 05_citator_graph | 0.4979__cit_abd_ot2_zst3_p__run3.zip | 0.4979 |
| 05_citator_graph | 0.4979__cit_abd_ot2_zst3_p__run4.zip | 0.4979 |
| 05_citator_graph | 0.4979__cit_abd_ot2_zst3_p__run1.zip | 0.4979 |
| 05_citator_graph | 0.4956__cit_abl_ot2_zst3_p__run1.zip | 0.4956 |
| 05_citator_graph | 0.4938__cit_ab_ot2_zst3_p__run2.zip | 0.4938 |
| 05_citator_graph | 0.4938__cit_ab_ot2_zst3_p__run3.zip | 0.4938 |
| 05_citator_graph | 0.4938__cit_ab_ot2_zst3_p__run1.zip | 0.4938 |
| 05_citator_graph | 0.4938__cit_al_ot2_zst3_p__run2.zip | 0.4938 |
