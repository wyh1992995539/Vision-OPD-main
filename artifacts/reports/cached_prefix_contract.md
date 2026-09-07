# Cached Prefix 6,241 Contract Audit

- Static status: **STATIC_CONTRACT_PASS**
- Runtime ablation status: **PENDING_TWO_GPU_PILOT**
- Cache: `/root/autodl-tmp/data/vision_opd_6241/cached_prefix_base_6241.parquet`
- Cache SHA256: `3a68ff1f8c7e63082f188dcc1bc508d8c0f9d7eafbcee748f00fd3769e6193ef`
- Records: `6241`
- Response token length: `2..1024`
- Finish reasons: `{"length": 48, "stop": 6193}`
- EOS appended after retokenization: `6193`
- Retokenized responses capped at 1,024: `1`

## Checks

- PASS — `cache_sha256_matches`
- PASS — `row_count_matches`
- PASS — `sample_ids_unique`
- PASS — `all_response_ids_roundtrip_exactly`
- PASS — `all_lengths_within_contract`
- PASS — `token_ids_source_frozen`
- PASS — `generation_hash_singleton`
- PASS — `base_model_path_singleton`
- PASS — `inference_errors_zero`
- PASS — `train_cache_sample_id_sets_equal`
- PASS — `all_train_prompts_and_student_images_bound`
- PASS — `generation_hash_matches_frozen_report`
- PASS — `generation_sampling_matches_online_rollout`
- PASS — `formal_data_contract_equal`
- PASS — `formal_actor_contract_equal`
- PASS — `formal_rollout_contract_equal`
- PASS — `formal_self_distillation_contract_equal`
- PASS — `formal_resource_contract_equal`
- PASS — `formal_training_contract_equal`
- PASS — `formal_model_train_template_paths_equal`
- PASS — `pilot_data_contract_equal`
- PASS — `pilot_actor_contract_equal`
- PASS — `pilot_rollout_contract_equal`
- PASS — `pilot_self_distillation_contract_equal`
- PASS — `pilot_resource_contract_equal`
- PASS — `pilot_training_contract_equal`
- PASS — `pilot_model_train_template_paths_equal`
- PASS — `pilot_frozen_sample_order_equal`
- PASS — `online_is_rollout_default`
- PASS — `trainer_has_separate_online_and_cached_dispatch`
- PASS — `cached_manager_skips_online_server_initialization`
- PASS — `cached_and_online_share_agent_loop_postprocess`

## Scope

The static audit proves that every cached response is bound to its frozen sample, prompt, Student image, Base tokenizer, generation contract, and model identity. It also proves that the online configuration remains the default and that the cached branch reports zero online generation calls in the tested dispatch path.

Strict prefix-source ablation remains pending until the 8-step cached Pilot runs on the same two-GPU resource contract and passes finite-loss, Student update, Teacher EMA, checkpoint, and cold-reload gates.
