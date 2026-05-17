"""Generate the (query, schema) corpus for the JSON-schema-adherence capability.

Produces ~250 prompts split into train.jsonl + eval.jsonl. Each row:

    {"id": "...", "query": "<natural-language task>", "schema": {<JSONSchema>}}

The generator deliberately spans the failure modes called out in capability.md:
nested objects, enum constraints, additionalProperties:false, format/pattern
constraints, anyOf discriminated unions, tuples, numeric ranges.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "datasets"
OUT_DIR.mkdir(exist_ok=True)
SEED = 4218


def domain_recipe_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "servings", "ingredients", "steps", "difficulty"],
        "properties": {
            "title": {"type": "string", "minLength": 3, "maxLength": 80},
            "servings": {"type": "integer", "minimum": 1, "maximum": 24},
            "ingredients": {
                "type": "array",
                "minItems": 2,
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["item", "quantity", "unit"],
                    "properties": {
                        "item": {"type": "string", "minLength": 2},
                        "quantity": {"type": "number", "exclusiveMinimum": 0},
                        "unit": {
                            "type": "string",
                            "enum": ["g", "kg", "ml", "l", "tsp", "tbsp", "cup", "piece"],
                        },
                    },
                },
            },
            "steps": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "string", "minLength": 10},
            },
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "prep_minutes": {"type": "integer", "minimum": 0},
        },
    }


def domain_event_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "start", "end", "venue", "tags"],
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "start": {"type": "string", "format": "date-time"},
            "end": {"type": "string", "format": "date-time"},
            "venue": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "city", "country"],
                "properties": {
                    "name": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string", "pattern": "^[A-Z]{2}$"},
                },
            },
            "tags": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": ["music", "tech", "art", "sports", "food", "talk"]},
                "uniqueItems": True,
            },
            "capacity": {"type": "integer", "minimum": 1},
        },
    }


def domain_bug_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "severity", "title", "repro", "tags", "status"],
        "properties": {
            "id": {"type": "string", "pattern": "^BUG-[0-9]{4}$"},
            "severity": {"type": "string", "enum": ["critical", "major", "minor", "trivial"]},
            "title": {"type": "string", "minLength": 5, "maxLength": 120},
            "repro": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "string", "minLength": 5},
            },
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "status": {"type": "string", "enum": ["open", "in_progress", "resolved"]},
            "owner": {"type": "string"},
        },
    }


def domain_book_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "authors", "isbn", "year", "genres"],
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "authors": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 2},
            },
            "isbn": {"type": "string", "pattern": "^[0-9]{3}-[0-9]{10}$"},
            "year": {"type": "integer", "minimum": 1450, "maximum": 2100},
            "genres": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": [
                    "fiction", "non-fiction", "fantasy", "sci-fi", "mystery",
                    "biography", "history", "science", "philosophy", "poetry",
                ]},
                "uniqueItems": True,
            },
            "page_count": {"type": "integer", "minimum": 1},
        },
    }


def domain_user_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["username", "email", "preferences", "joined"],
        "properties": {
            "username": {"type": "string", "pattern": "^[a-z0-9_]{3,16}$"},
            "email": {"type": "string", "format": "email"},
            "preferences": {
                "type": "object",
                "additionalProperties": False,
                "required": ["theme", "notifications"],
                "properties": {
                    "theme": {"type": "string", "enum": ["light", "dark", "auto"]},
                    "notifications": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["email", "push"],
                        "properties": {
                            "email": {"type": "boolean"},
                            "push": {"type": "boolean"},
                            "sms": {"type": "boolean"},
                        },
                    },
                    "language": {"type": "string", "pattern": "^[a-z]{2}(-[A-Z]{2})?$"},
                },
            },
            "joined": {"type": "string", "format": "date"},
            "verified": {"type": "boolean"},
        },
    }


def domain_invoice_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["invoice_id", "customer", "line_items", "total", "currency"],
        "properties": {
            "invoice_id": {"type": "string", "pattern": "^INV-[0-9]{6}$"},
            "customer": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "email"],
                "properties": {
                    "name": {"type": "string", "minLength": 2},
                    "email": {"type": "string", "format": "email"},
                    "vat": {"type": "string"},
                },
            },
            "line_items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sku", "qty", "unit_price"],
                    "properties": {
                        "sku": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{3,5}$"},
                        "qty": {"type": "integer", "minimum": 1},
                        "unit_price": {"type": "number", "exclusiveMinimum": 0},
                    },
                },
            },
            "total": {"type": "number", "minimum": 0},
            "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY", "CAD"]},
            "issued": {"type": "string", "format": "date"},
        },
    }


def domain_task_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "title", "priority", "tags"],
        "properties": {
            "id": {"type": "string", "pattern": "^T-[A-Z0-9]{5,8}$"},
            "title": {"type": "string", "minLength": 4, "maxLength": 120},
            "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
            "tags": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 2},
                "uniqueItems": True,
            },
            "due": {"type": "string", "format": "date"},
            "estimate_hours": {"type": "number", "minimum": 0.25, "maximum": 80},
        },
    }


def domain_endpoint_schema() -> dict:
    """anyOf discriminated union: REST endpoint definitions vary by method."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "spec"],
        "properties": {
            "path": {"type": "string", "pattern": "^/[a-z0-9_/{}-]+$"},
            "spec": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["method", "responses"],
                        "properties": {
                            "method": {"const": "GET"},
                            "responses": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "integer", "minimum": 100, "maximum": 599},
                            },
                            "query_params": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["method", "body_schema", "responses"],
                        "properties": {
                            "method": {"enum": ["POST", "PUT", "PATCH"]},
                            "body_schema": {"type": "string"},
                            "responses": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "integer", "minimum": 100, "maximum": 599},
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["method", "soft_delete"],
                        "properties": {
                            "method": {"const": "DELETE"},
                            "soft_delete": {"type": "boolean"},
                        },
                    },
                ]
            },
        },
    }


def domain_coordinate_schema() -> dict:
    """prefixItems — tuple-style array constraint."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["place", "coordinates", "elevation_meters"],
        "properties": {
            "place": {"type": "string"},
            "coordinates": {
                "type": "array",
                "prefixItems": [
                    {"type": "number", "minimum": -90, "maximum": 90},
                    {"type": "number", "minimum": -180, "maximum": 180},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "elevation_meters": {"type": "integer"},
        },
    }


def domain_pet_schema() -> dict:
    """oneOf discriminated by 'kind' const."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "kind", "details"],
        "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["dog", "cat", "bird"]},
            "details": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["breed", "good_with_kids"],
                        "properties": {
                            "breed": {"type": "string"},
                            "good_with_kids": {"type": "boolean"},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["indoor", "color"],
                        "properties": {
                            "indoor": {"type": "boolean"},
                            "color": {"type": "string"},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["species", "wing_span_cm"],
                        "properties": {
                            "species": {"type": "string"},
                            "wing_span_cm": {"type": "number", "minimum": 1},
                        },
                    },
                ]
            },
        },
    }


# (schema_builder, query_template, fillers) tuples
# query_template uses {hint} for variation
DOMAINS = [
    (
        domain_recipe_schema,
        "Create a recipe for {hint}. Output as JSON conforming to the schema.",
        ["a vegetarian chili", "homemade pasta carbonara", "a simple chocolate cake",
         "kung pao chicken", "miso soup", "beef bourguignon", "Thai green curry",
         "tabbouleh salad", "lemon meringue pie", "shakshuka", "ratatouille",
         "tom yum soup", "biryani for 4", "borscht", "okonomiyaki",
         "pho ga", "khachapuri", "gazpacho", "moussaka", "pierogi",
         "a vegan banana bread", "an Iranian fesenjan", "Cuban ropa vieja",
         "Korean japchae", "a low-carb cauliflower curry"],
    ),
    (
        domain_event_schema,
        "Describe an upcoming event: {hint}. Return as JSON.",
        ["a jazz festival in Tokyo this October", "a Python conference in Berlin",
         "a marathon in Boston", "an art opening in Mexico City", "a tech summit in Seoul",
         "a wine tasting in Lyon", "a food truck rally in Austin",
         "a poetry reading in Edinburgh", "a chess tournament in Reykjavik",
         "a hackathon in Bangalore", "a coffee championship in Melbourne",
         "a startup pitch night in Lagos", "a chamber-music concert in Vienna",
         "a film screening in Marrakech", "a sustainability summit in Stockholm",
         "a robotics demo in Shenzhen", "a board-game convention in Essen",
         "a UX meetup in São Paulo", "a typography workshop in Amsterdam",
         "a yoga retreat in Bali", "a craft-beer fest in Prague",
         "a developer evening in Tel Aviv", "a science-fiction con in Helsinki",
         "a culinary expo in Mumbai", "a literature festival in Cartagena"],
    ),
    (
        domain_bug_schema,
        "File a bug report for the following issue: {hint}",
        ["the iOS app crashes when scanning a barcode in low light",
         "search results return duplicates after a stale cache hit",
         "the export-to-PDF button does nothing in Safari",
         "dark mode flashes white during page transitions",
         "uploading files larger than 10MB silently fails",
         "OAuth callback hangs for 30 seconds before timing out",
         "the cart total ignores promo codes for guest users",
         "websocket reconnect loops forever on flaky networks",
         "timezone conversion is off by one hour for users in Lord Howe Island",
         "drag-and-drop in the kanban board occasionally swaps wrong cards",
         "PDF preview shows blank pages for documents with embedded fonts",
         "the SSO logout button leaves session cookies behind",
         "the notification badge stays at 1 after all notifications are read",
         "the calendar view drops events at DST transitions",
         "Stripe webhooks 500 when refund metadata is empty",
         "iOS push tokens are not rotated when a device is restored from backup",
         "the rate-limit header is missing on free-tier responses",
         "Markdown rendering escapes backticks inside code fences",
         "the API returns 200 with empty body when a query exceeds 100K characters",
         "uploading EXIF-rotated photos shows them sideways in the gallery",
         "the wishlist export omits items added in the last 24 hours",
         "the date-range picker accepts an end-date before the start",
         "double-tapping the like button registers as a like-unlike pair",
         "the unsubscribe link from welcome emails 404s after 7 days",
         "the search facet count is off by one when a filter is applied"],
    ),
    (
        domain_book_schema,
        "Describe a book: {hint}. Output as JSON.",
        ["a sweeping fantasy epic about a cartographer who maps memories",
         "a non-fiction guide to coastal geology",
         "a mystery set in a 1920s Parisian opera house",
         "a memoir of growing up in rural Bhutan",
         "a hard science-fiction novel about asteroid mining",
         "a history of the printing press",
         "a philosophy treatise on attention and boredom",
         "a poetry collection on grief and gardening",
         "a biography of an unsung scientist",
         "a fictional account of the first Mars colony",
         "a cookbook organised by mood",
         "a thriller about a maritime smuggler",
         "a children's adventure across a magical archipelago",
         "a historical novel set during the fall of Byzantium",
         "a literary anthology of letters between two strangers",
         "a science-popular book on octopus cognition",
         "a sweeping family saga across three generations of beekeepers",
         "a graphic novel about a librarian on the moon",
         "a manual on slow-fermented bread",
         "a thriller set inside an Antarctic research station",
         "an Afrofuturist novel about ancestor memory",
         "a deep dive into the history of cartographic projections",
         "a magical-realist novel set in a coffee plantation",
         "a YA novel about a swim team in a desert town",
         "a meditation on long-form letter writing"],
    ),
    (
        domain_user_schema,
        "Create a user profile based on this description: {hint}",
        ["a UX designer who loves cycling and prefers a dark UI",
         "a high-school physics teacher in Ireland",
         "a competitive Scrabble player in Nairobi",
         "a maritime engineer from Stavanger",
         "a freelance illustrator in Buenos Aires",
         "a wine importer in Vancouver",
         "a Buddhist nun who occasionally tweets about cosmology",
         "a paleontologist working from Patagonia",
         "an opera-singer-turned-programmer in Naples",
         "a podiatrist who collects vintage telephones",
         "a SwiftUI developer who runs ultramarathons",
         "a beekeeper in rural Slovenia",
         "a yoga instructor in Reykjavik",
         "a database administrator who breeds koi",
         "an indie game developer in Lagos",
         "a typeface designer in Berlin",
         "a marine biologist studying Hawaiian reefs",
         "a perfumer based outside Grasse",
         "a backcountry pilot in the Yukon",
         "a textile conservator working from Lyon",
         "a competitive memory athlete from Mongolia",
         "a freelance translator working between Korean and Portuguese",
         "a sommelier in Sydney",
         "a lighthouse keeper turned podcaster",
         "a children's-book illustrator from Hiroshima"],
    ),
    (
        domain_invoice_schema,
        "Generate an invoice for: {hint}",
        ["3 design consults at $150/hr each, billed in USD",
         "a one-month enterprise SaaS subscription in EUR",
         "wholesale order of 240 ceramic mugs to a coffee chain in GBP",
         "freelance code review of a JS bundle at $200/hr, USD",
         "two-day photography shoot with prints, billed in CAD",
         "custom furniture commission with installation, EUR",
         "monthly accounting services for a small bakery, GBP",
         "annual licence for a CAD plugin, USD",
         "translation services for a 30-page contract, JPY",
         "five blog posts at $300 each, USD",
         "a quarterly cloud-hosting retainer, USD",
         "a guided wildlife tour for 8 guests, USD",
         "a podcast-editing package of 4 episodes, EUR",
         "annual subscription to a small SaaS, billed in GBP",
         "a wedding-cake order with delivery, EUR",
         "5 hours of legal advice at €250/hr",
         "monthly SEO retainer billed in CAD",
         "a one-time data-recovery service in JPY",
         "a 12-month payroll subscription for 30 employees, USD",
         "a custom logo package with three rounds of revisions, GBP",
         "a private cooking class for 6 people, EUR",
         "an annual security audit, USD",
         "monthly retainer for executive coaching, EUR",
         "a fixed-bid copywriting project for a startup, USD",
         "an SEO content-audit deliverable, GBP"],
    ),
    (
        domain_task_schema,
        "Create a task with these details: {hint}",
        ["build the unsubscribe flow for the email digest",
         "investigate flaky CI test failures on the macOS runner",
         "rotate the production database credentials",
         "draft the Q3 launch announcement post",
         "audit the AWS bill for unused volumes",
         "migrate the staging environment to the new VPC",
         "fix the off-by-one bug in the pagination cursor",
         "spec out the new permissions UI for admins",
         "set up automated dependency-bot reviews",
         "write a postmortem for the Tuesday outage",
         "consolidate the four Slack notification channels into one",
         "interview three candidates for the senior backend role",
         "ship the SOC 2 evidence checklist",
         "decommission the legacy reporting service",
         "review and approve the new privacy policy",
         "design the new onboarding email sequence",
         "set up uptime monitoring on the public API",
         "research alternatives to our analytics vendor",
         "publish the quarterly transparency report",
         "fix the broken dark-mode contrast on the dashboard",
         "set up a quarterly chaos-engineering exercise",
         "harden the cookie security headers",
         "consolidate two duplicate CI workflows",
         "stand up a load test against the new auth service",
         "draft the disaster-recovery runbook for the new region"],
    ),
    (
        domain_endpoint_schema,
        "Define an API endpoint: {hint}. Return as JSON.",
        ["GET /users/{id} returning 200, 404",
         "POST /orders with a JSON body, returning 201, 400, 409",
         "DELETE /sessions/{token} with soft-delete enabled",
         "GET /search with query_params q, page, returning 200, 400",
         "PUT /products/{sku} with a JSON body, returning 200, 404, 422",
         "DELETE /tokens/{id} with soft-delete disabled",
         "PATCH /users/{id}/profile with a JSON body, returning 200, 400, 409",
         "GET /health returning only 200",
         "POST /webhooks with a JSON body, returning 202, 400",
         "GET /reports/{year}/{month} returning 200, 403, 404",
         "DELETE /comments/{id} with soft-delete enabled",
         "GET /users with query_params limit, offset, sort, returning 200, 400",
         "POST /uploads with a JSON body, returning 201, 413, 415",
         "PUT /settings/{key} with a JSON body, returning 204, 400",
         "DELETE /branches/{name} with soft-delete disabled",
         "PATCH /tasks/{id} with a JSON body, returning 200, 404, 422",
         "GET /metrics with query_params from, to, returning 200, 400, 401",
         "POST /invitations with a JSON body, returning 201, 400, 409",
         "DELETE /devices/{id} with soft-delete enabled",
         "GET /audit-log with query_params actor, since, returning 200, 401"],
    ),
    (
        domain_coordinate_schema,
        "Describe the location of {hint}",
        ["the summit of Aconcagua",
         "the Burj Khalifa",
         "the geographic centre of Australia",
         "the entrance to the Mariana Trench",
         "Cape Point in South Africa",
         "the South Pole",
         "the deepest point of Lake Baikal",
         "the Statue of Liberty",
         "the McMurdo Station in Antarctica",
         "the centre of Times Square",
         "the summit of Mount Fuji",
         "the Eiffel Tower",
         "the headwaters of the Amazon",
         "Lake Titicaca",
         "the centre of Tiananmen Square",
         "Brandenburg Gate",
         "Galápagos Islands",
         "the Niagara Falls (Canadian side)",
         "Patagonia's Perito Moreno glacier",
         "the Hoover Dam"],
    ),
    (
        domain_pet_schema,
        "Describe a pet: {hint}",
        ["a friendly Labrador named Biscuit, good with children",
         "an indoor black cat called Pepper",
         "a Siberian husky named Astrid, somewhat skittish around kids",
         "a tabby cat named Inks, lives outside",
         "a budgerigar named Citron",
         "a corgi puppy named Doris, very kid-friendly",
         "a Persian cat called Marble, strictly indoors",
         "a cockatiel named Pip with a 30cm wingspan",
         "a German shepherd named Olympia, kid-friendly",
         "a calico cat named Tessellate, indoor only",
         "a hyacinth macaw named Bluebottle",
         "a beagle puppy named Toaster, good with kids",
         "an indoor Maine coon named Tribble",
         "a barn owl named Sift",
         "a poodle named Sorrel, mediocre with kids",
         "a Bengal cat called Curfew, indoors",
         "an Eurasian eagle-owl named Vesper",
         "a dachshund named Strudel, kid-friendly",
         "a Sphynx cat named Latitude, strictly indoor",
         "a kestrel named Whetstone"],
    ),
]


def main() -> None:
    random.seed(SEED)
    rows: list[dict] = []
    domain_index = 0
    for builder, template, fillers in DOMAINS:
        schema = builder()
        for hint in fillers:
            rows.append({
                "id": f"{builder.__name__}_{domain_index:04d}",
                "query": template.format(hint=hint),
                "schema": schema,
            })
            domain_index += 1
    print(f"built {len(rows)} prompts across {len(DOMAINS)} domains")
    random.shuffle(rows)

    # 80/20 split. With 250 prompts that's 200 train / 50 eval.
    split = int(len(rows) * 0.8)
    train, eval_set = rows[:split], rows[split:]
    print(f"split: {len(train)} train, {len(eval_set)} eval")

    with (OUT_DIR / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with (OUT_DIR / "eval.jsonl").open("w") as f:
        for r in eval_set:
            f.write(json.dumps(r) + "\n")
    print(f"wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
