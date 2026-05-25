import { appendFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const TEMPERATURE = 0.6;
const TOP_P = 0.95;

export default function (pi: ExtensionAPI) {
	pi.on("before_provider_request", (event) => {
		if (process.env.KILN_PI_PAYLOAD_TRACE === "1") {
			appendFileSync(
				".pi-provider-payload.jsonl",
				`${JSON.stringify({
					temperature: TEMPERATURE,
					top_p: TOP_P,
					original_temperature: event.payload.temperature,
					original_top_p: event.payload.top_p,
				})}\n`,
				"utf8",
			);
		}
		return { ...event.payload, temperature: TEMPERATURE, top_p: TOP_P };
	});
}
