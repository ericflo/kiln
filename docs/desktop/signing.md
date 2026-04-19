# macOS Code Signing and Notarization

Downloaded `.dmg`s must be signed with a Developer ID Application certificate
and notarized by Apple for Gatekeeper to accept them on a fresh install.
Without signing, users see "Kiln Desktop is damaged and can't be opened"
the first time they try to open the downloaded bundle.

The
[`desktop-build.yml`](../../.github/workflows/desktop-build.yml) workflow
handles signing, notarization (via `notarytool`), and stapling automatically
when the `APPLE_*` secrets below are present; tauri-action creates a temp
keychain on the macOS runner, imports the `.p12`, signs with the hardened
runtime, and submits to notarytool. A follow-up step notarizes and staples
the `.dmg` wrapper itself.

## Required Secrets

The following GitHub Actions secrets must be set on the `ericflo/kiln`
repository for macOS release builds to ship signed, notarized `.dmg`s:

| Secret                                | Value                                                                |
|---------------------------------------|----------------------------------------------------------------------|
| `APPLE_CERTIFICATE`                   | Base64 of the Developer ID Application `.p12`.                       |
| `APPLE_CERTIFICATE_PASSWORD`          | Password used when exporting the `.p12` (no trailing newline).       |
| `APPLE_SIGNING_IDENTITY`              | e.g. `Developer ID Application: Eric Florenzano (F6ZGE4FAML)`.       |
| `APPLE_API_ISSUER`                    | App Store Connect API key issuer UUID.                               |
| `APPLE_API_KEY`                       | App Store Connect API key ID (10 chars).                             |
| `APPLE_API_KEY_BASE64`                | Base64 of the `AuthKey_XXXX.p8` downloaded from App Store Connect.   |

Without the `APPLE_*` secrets the macOS build still succeeds but downloaded
`.dmg`s trip Gatekeeper on first open.

## First-Time Setup

Per Apple Developer Program seat — already completed for this repo, included
here for continuity and future rotation.

1. Generate a Certificate Signing Request and keep the matching private
   key — the `.p12` is this key plus the cert Apple issues:
   ```bash
   mkdir -p ~/apple-signing-setup && cd ~/apple-signing-setup
   openssl genrsa -out developerID.key 2048
   openssl req -new -key developerID.key -out developerID.csr \
     -subj "/emailAddress=YOUR_EMAIL/CN=YOUR NAME/C=US"
   ```
2. At https://developer.apple.com/account/resources/certificates create
   a new **Developer ID Application** certificate, upload `developerID.csr`,
   and download the resulting `.cer`.
3. Bundle the cert and key into a `.p12` — **must** use `-legacy`, since
   macOS `security import` on the CI runner cannot read OpenSSL 3's default
   modern PKCS12 format and fails with a confusing "MAC verification failed"
   password error:
   ```bash
   openssl x509 -inform DER -in developerID.cer -out developerID.pem
   openssl pkcs12 -export -legacy \
     -inkey developerID.key -in developerID.pem \
     -out developerID.p12 -name "Developer ID Application" \
     -passout pass:YOUR_P12_PASSWORD
   base64 -i developerID.p12 | pbcopy   # → APPLE_CERTIFICATE
   ```
4. Create an **App Store Connect API key** at
   https://appstoreconnect.apple.com/access/integrations/api with the
   "Developer" role. Download the `.p8` (only available once — save it).
   Note the key ID (10 chars) and the issuer UUID:
   ```bash
   base64 -i AuthKey_XXXXXX.p8 | pbcopy   # → APPLE_API_KEY_BASE64
   ```
5. Upload all six secrets to the repo:
   ```bash
   gh secret set APPLE_CERTIFICATE --repo ericflo/kiln < <(base64 -i developerID.p12)
   gh secret set APPLE_CERTIFICATE_PASSWORD --repo ericflo/kiln --body 'YOUR_P12_PASSWORD'
   gh secret set APPLE_SIGNING_IDENTITY --repo ericflo/kiln --body 'Developer ID Application: YOUR NAME (TEAMID)'
   gh secret set APPLE_API_ISSUER --repo ericflo/kiln --body 'ISSUER_UUID'
   gh secret set APPLE_API_KEY --repo ericflo/kiln --body 'KEY_ID'
   gh secret set APPLE_API_KEY_BASE64 --repo ericflo/kiln < <(base64 -i AuthKey_XXXXXX.p8)
   ```

The Developer ID certificate is valid for five years; renew before it
expires. The App Store Connect key can be rotated at any time — revoke
the old one in the portal and update the three `APPLE_API_*` secrets.

## Note on `.dmg` Notarization

tauri-action notarizes the `.app` bundle but **not** the `.dmg` wrapper,
so the workflow has a follow-up `xcrun notarytool submit` + `xcrun stapler
staple` step that runs against the `.dmg` and re-uploads it to the draft
release with `gh release upload --clobber`. Without this step the
downloaded `.dmg` would still trip Gatekeeper on first open as
"Unnotarized Developer ID" even though the `.app` inside is fully
notarized.

Both macOS signing steps are guarded on
`runner.os == 'macOS' && startsWith(github.ref, 'refs/tags/desktop-v')`
so they only run when building for release (i.e. on a `desktop-v*` tag).
A non-tag push has no release to upload to.

## Verifying a Release

After CI finishes, download the `.dmg` from the draft release and check
Gatekeeper will accept it as though the user had just downloaded it in a
browser:

```bash
cd /tmp
curl -fsSL -o v.dmg <release_url>
# Spoof the Safari/Chrome quarantine xattr so Gatekeeper does a full check.
xattr -w com.apple.quarantine "0083;$(printf %x $(date +%s));Chrome;" v.dmg
spctl -a -vv -t open --context context:primary-signature v.dmg
```

Expected: `accepted` with `source=Notarized Developer ID`.
