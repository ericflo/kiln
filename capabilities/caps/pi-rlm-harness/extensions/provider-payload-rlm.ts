import { appendFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MAX_TOKENS = 4096;
const TEMPERATURE = 0.0;
const TOP_P = 1.0;

export default function (pi: ExtensionAPI) {
	pi.on("before_provider_request", (event) => {
		const payload = {
			...event.payload,
			max_tokens: MAX_TOKENS,
			temperature: TEMPERATURE,
			top_p: TOP_P,
			metadata: {
				...(event.payload.metadata ?? {}),
				kiln_rlm_harness: true,
				kiln_rlm_window: {
					prefix_tokens: 8192,
					input_tokens: 4096,
					output_tokens: 4096,
				},
			},
		};

		if (process.env.KILN_PI_RLM_PAYLOAD_TRACE === "1") {
			appendFileSync(
				".pi-rlm-provider-payload.jsonl",
				`${JSON.stringify({
					max_tokens: payload.max_tokens,
					temperature: payload.temperature,
					top_p: payload.top_p,
					message_count: Array.isArray(payload.messages)
						? payload.messages.length
						: null,
					tool_count: Array.isArray(payload.tools) ? payload.tools.length : null,
				})}\n`,
				"utf8",
			);
		}

		return payload;
	});
}
