## Per-pair summary

| pair | ref_top1 | kiln_top1 | kiln_matches_ref | notes |
|------|----------|-----------|------------------|-------|
| seed42 | 3074 | 3074 | True | draft_token_id=3074 mtp_pos=2 base_pos=553 |
| seed43 | 16078 | 16078 | True | draft_token_id=16078 mtp_pos=2 base_pos=527 |
| seed44 | 1814 | 1814 | True | draft_token_id=1814 mtp_pos=2 base_pos=523 |

## Per-tap splice flips (kiln bf16 → fp32 reference)

| tap | flip@seed42 | flip@seed43 | flip@seed44 | total_flips |
|-----|-----------|-----------|-----------|-------------|
| tok_embed | . | . | . | 0 |
| norm_emb | . | . | . | 0 |
| norm_h | . | . | . | 0 |
| fc_input | . | . | . | 0 |
| fc_output | . | . | . | 0 |
| post_pre_attn_norm | . | . | . | 0 |
| post_q_proj_raw | . | . | . | 0 |
| post_k_proj | . | . | . | 0 |
| post_v_proj | . | . | . | 0 |
| post_q_split | . | . | . | 0 |
| post_gate_split | . | . | . | 0 |
| post_q_norm | . | . | . | 0 |
| post_k_norm | . | . | . | 0 |
| post_q_rope | . | . | . | 0 |
| post_k_rope | . | . | . | 0 |
| attn_out | . | . | . | 0 |
| post_attn_gated | . | . | . | 0 |
| post_o_proj | . | . | . | 0 |
| post_attn_residual | . | . | . | 0 |
| post_pre_mlp_norm | . | . | . | 0 |
| post_mlp | . | . | . | 0 |
| post_layer | . | . | . | 0 |
| post_final_ln | . | . | . | 0 |
| mtp_logits | . | . | . | 0 |

## Per-tap ref_top1 margin after splice (fp32-logit units)

| tap | margin@seed42 | margin@seed43 | margin@seed44 |
|-----|-------------|-------------|-------------|
| tok_embed | +8.1824 | +3.1893 | +7.2255 |
| norm_emb | +8.1835 | +3.1847 | +7.2250 |
| norm_h | +8.1864 | +3.1880 | +7.2181 |
| fc_input | +8.1876 | +3.1833 | +7.2177 |
| fc_output | +8.1908 | +3.1819 | +7.2299 |
| post_pre_attn_norm | +8.1914 | +3.1841 | +7.2273 |
| post_q_proj_raw | +8.1797 | +3.1951 | +7.2293 |
| post_k_proj | +8.1824 | +3.1893 | +7.2255 |
| post_v_proj | +8.1745 | +3.1749 | +7.2327 |
| post_q_split | +8.1824 | +3.1893 | +7.2255 |
| post_gate_split | +8.1797 | +3.1951 | +7.2293 |
| post_q_norm | +8.1824 | +3.1893 | +7.2255 |
| post_k_norm | +8.1824 | +3.1893 | +7.2255 |
| post_q_rope | +8.1824 | +3.1893 | +7.2255 |
| post_k_rope | +8.1824 | +3.1893 | +7.2255 |
| attn_out | +8.1745 | +3.1749 | +7.2327 |
| post_attn_gated | +8.1438 | +3.1836 | +7.2423 |
| post_o_proj | +8.1428 | +3.1861 | +7.2432 |
| post_attn_residual | +8.1566 | +3.1780 | +7.2444 |
| post_pre_mlp_norm | +8.1620 | +3.1812 | +7.2336 |
| post_mlp | +8.1845 | +3.1773 | +7.2330 |
| post_layer | +8.1894 | +3.1707 | +7.2284 |
| post_final_ln | +8.1868 | +3.1665 | +7.2193 |
| mtp_logits | +8.2500 | +3.1250 | +7.2500 |

