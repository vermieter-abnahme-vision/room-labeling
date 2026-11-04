# Room Labeling

Mobile-first Tool, mit dem Freunde Frames aus Wohnungsvideos nach Raumtyp labeln
(Bad · Küche · Wohnzimmer · Schlafzimmer · Flur · Unklar). Das Ergebnis ist ein
**reproduzierbares, leakage-freies Datapack** (JSONL-Manifest), das 1:1 in ein ML-Projekt
übernommen werden kann.

> ## ⚠️ Daten bleiben lokal – nie pushen
> Dieses Repo ist **öffentlich** und enthält **nur Code**. Wohnungsfotos, Videos, die SQLite-DB,
> Embeddings, Modelle, Exporte und `.env` liegen ausschließlich auf deinem Rechner bzw. dem VPS
> (Docker-Volumes) und sind per `.gitignore`, Pre-Commit-Hook (`no-private-data`) und
> CI-Secret-Scan (gitleaks) gegen versehentliches Committen abgesichert.

## Inhalt

- [Architektur](#architektur) · [Quickstart](#quickstart) · [OIDC-Konfiguration](#oidc-konfiguration-authentik-beispiel)
- [Workflow Ingest → Embed → Train → Label → Export](#workflow)
- [Datenmodell & Konsens](#datenmodell--konsens) · [Datapack](#datapack-export)
- [Security Model](#security-model) · [Betrieb](#betrieb-docker) · [Repo-Settings](#github-repo-einstellungen-manuell)

## Architektur

```
app/
  config.py        pydantic-settings (alles aus ENV)
  db.py            Engine, WAL-Pragmas, Lese-/Schreib-Sessions (BEGIN IMMEDIATE)
  models.py        SQLAlchemy-2.x-Modelle (images, labelers, annotations, consensus, model_predictions)
  labelschemes.py  Scheme-Definitionen (single/multi, erlaubte Labels) – zentral
  schemas.py       Pydantic-DTOs
  security.py      CSP/Header-Middleware, CSRF, Redirect-Sicherheit, Rate-Limiter
  deps.py          Auth-/Admin-Dependencies
  services/        gesamte Geschäftslogik (kein FastAPI-Import!)
    annotations    idempotentes Submit  ·  consensus  Konsens/Konflikt/Admin-Entscheid
    sampling       nächstes Bild (Active Learning)  ·  stats  Agreement (Krippendorff-α, Cohen-κ)
    ingest         Ordner / Video (ffmpeg) / Library  ·  export  Manifest, Splits, Imagefolder
    inference      SigLIP-Embeddings, Zero-Shot, LogisticRegression-Probe
  routers/         dünne HTTP-Schicht (auth, labeling, images, admin, health)
  web/             Jinja2-Templates + Vanilla-JS (kein Inline-JS, strikte CSP)
scripts/           CLI-Einstiege (nutzen dieselben Services)
alembic/           Migrationen        tests/   pytest (Konsens, Sampling, Export, Ingest, Web/Security …)
```

Schichtregel: Router → Services → Models. Services kennen weder FastAPI noch Templates.

## Quickstart

Voraussetzungen: [uv](https://docs.astral.sh/uv/), Python 3.12, `ffmpeg` (nur für Video-Ingest).

```bash
cp .env.example .env            # SECRET_KEY + OIDC_* ausfüllen (siehe unten)
make install                    # uv sync --extra probe --extra siglip
make migrate                    # Alembic-Migrationen
make dev                        # http://localhost:8000
```

`SECRET_KEY` erzeugen (≥ 32 Byte, zufällig):

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Nur Tests/Lint, ohne schwere ML-Pakete: `uv sync --extra probe && make check`.

## OIDC-Konfiguration (Authentik-Beispiel)

Provider-agnostisch: Authorization-Code-Flow **mit PKCE (S256)**, `state` und `nonce`; die
ID-Token-Signatur (JWKS), `iss`, `aud`, `exp` werden geprüft. Alles per ENV:

| Variable | Bedeutung |
|---|---|
| `OIDC_ISSUER` | Issuer-URL; Discovery unter `<issuer>/.well-known/openid-configuration` |
| `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | Client-Credentials des Providers |
| `OIDC_REDIRECT_URI` | exakt die beim Provider registrierte Callback-URL, z. B. `https://labels.example.com/auth/callback` |
| `OIDC_ALLOWED_REDIRECT_URIS` | optionale Allowlist weiterer URIs (`OIDC_REDIRECT_URI` muss enthalten sein) |
| `OIDC_SCOPES` | Standard `openid email profile` (`openid` ist Pflicht) |
| `ADMIN_SUBS` / `ADMIN_EMAILS` | Admin-Allowlist (OIDC-`sub` oder **verifizierte** E-Mail) |

**Authentik:**

1. *Applications → Providers → Create → OAuth2/OpenID Provider*; Client-Typ **Confidential**.
2. *Redirect URIs/Origins*: `https://labels.example.com/auth/callback` (Modus *Strict*). Scopes: `openid`, `email`, `profile`.
3. *Application* anlegen, Provider zuweisen, Zugriff per Gruppe/Policy auf deine Labeler beschränken.
4. `.env`:
   ```
   OIDC_ISSUER=https://auth.example.com/application/o/<application-slug>/
   OIDC_CLIENT_ID=…
   OIDC_CLIENT_SECRET=…
   OIDC_REDIRECT_URI=https://labels.example.com/auth/callback
   ADMIN_SUBS=<dein-sub>      # steht im ID-Token / unter Authentik → Users
   ```

Andere Provider (Keycloak `https://host/realms/<realm>`, Google `https://accounts.google.com`, …)
funktionieren mit denselben Variablen. **Wer sich anmelden darf, entscheidet der Provider** – die
App legt jeden authentifizierten Nutzer als Labeler an. Beschränke den Zugriff deshalb dort.

## Workflow

```
Ingest ──► Embed ──► Train/Zero-Shot ──► Label (Mobile) ──► Export ──► ML-Projekt
```

1. **Ingest** – Material in `./images_in/<wohnung>/*.jpg` bzw. `./videos_in/<wohnung>.mp4` legen:
   ```bash
   make ingest        # ffmpeg-Frames mit VIDEO_FPS (Default 1), sha256-Dedup, group_id gesetzt
   ```
   `group_id` = Unterordnername bzw. Videodateiname (Stem) → Grundlage für den Split ohne Leakage.
   Optional vorlabelte Library-Bilder (Pseudo-Labeler `library`, final, nicht in Agreement-Stats):
   ```bash
   make import-library DATASET=places365 ROOT=./library_data ARGS="--download --max-per-class 300"
   make import-library DATASET=indoor67  ROOT=./indoorCVPR_09      # manuell herunterladen (nur http)
   ```
   Das Mapping Szenenklasse → Raumlabel steht in `scripts/import_library.py`; nicht zuordenbare
   Klassen werden übersprungen.
2. **Embed** – `make embed` berechnet SigLIP-Embeddings (`google/siglip-base-patch16-224`),
   gecached als `.npy` (Key = sha256). Für reproduzierbare Embeddings `SIGLIP_REVISION` pinnen.
3. **Train** – `make train` trainiert eine LogisticRegression auf allen finalen Labels (inkl.
   Library), meldet die **gruppenweise** Cross-Validation-Accuracy und schreibt Vorhersagen für
   *alle* Bilder in `model_predictions`. Fehlen Daten, greift automatisch der **Zero-Shot-Fallback**
   (`make zeroshot` erzwingt ihn). Mit `--include-unfinalized` fließen auch einstimmige Einzel-Labels
   noch nicht finaler Bilder ein (Cold-Start).
4. **Label** – Handy öffnen, per OIDC anmelden. Ein Bild groß, Vorschlag als vorausgewählter Button
   („Vorschlag: Küche – bestätigen oder korrigieren“, 1 Tap), Tasten `1–6`, `Enter`, `S`. Nach jedem
   Label sofort das nächste Bild; „Zurück“ korrigiert das letzte Label. Nach jedem Trainingslauf
   bevorzugt das Sampling die unsichersten Bilder (Active Learning) – Schritte 3→4 beliebig wiederholen.
5. **Export** – `/admin/export` oder `make export` → `manifest.jsonl` + `meta.json`.
   `make materialize MANIFEST=var/exports/<stamp>/manifest.jsonl` erzeugt `data/<split>/<label>/<sha>.jpg`
   (Hardlinks). Das Manifest bleibt führend.

## Datenmodell & Konsens

Annotationen sind generisch über `scheme` (Default `room`), damit später Element-Labels
(Fenster, Heizung …) **ohne Umbau** möglich sind: neues `LabelScheme(kind=MULTI, …)` in
`app/labelschemes.py` registrieren, mehrere Rows pro (Bild, Labeler, Scheme) sind erlaubt.

- `images` (sha256 unique, relativer Pfad, `source`, `group_id`, `extra` JSON)
- `labelers` · `annotations` (unique image/labeler/scheme/label) · `consensus` · `model_predictions`
  (+ `margin` für Top-2-Margin-Sampling; `consensus.resolved_by_id` markiert Admin-Entscheide)
- Label-Set per `ROOM_LABELS` erweiterbar (lowercase `[a-z0-9_]`, Reihenfolge = Button/Shortcut).

Konsens (`services/consensus.py`): 2 übereinstimmende Labels → `final` (agreement 1.0);
Dissens → `needs_more` (3. Label), Mehrheit entscheidet; 3-fach-Split → `conflict` →
Admin-Queue. `unklar` ist ein normales Label. Re-Submit desselben Labelers ist ein Update.
Library-Labels sind vertrauenswürdig und finalisieren sofort; Admin-Entscheide sind „sticky“.

Sampling (`services/sampling.py`): nur ungelabelt für dich & nicht `final`/`conflict`;
`needs_more` vor `open`; darin unsicherstes zuerst (Margin oder niedrigste Confidence,
`SAMPLING_STRATEGY`), ohne Prediction ans Ende (stabile Pseudo-Zufallsreihenfolge); aus dem
Pool der unsichersten wird **klassen-rotierend** gewählt – deterministisch pro (Labeler,
Fortschritt), also resume-sicher ohne Bias auf eine Klasse.

> Hinweis: Der sichtbare Modellvorschlag kann Labeler beeinflussen (Anchoring) und das
> Agreement beschönigen. Für eine „unabhängige“ Qualitätsmessung ggf. temporär ohne
> Vorschläge labeln lassen (`model_predictions` leeren bzw. kein Training).

## Datapack-Export

`manifest.jsonl` (Quelle der Wahrheit), eine Zeile pro finalem Bild:

```json
{"sha256":"…","path":"ab/ab12….jpg","group_id":"wohnung1","source":"upload_video","scheme":"room",
 "final_label":"kueche","agreement":1.0,"resolution":"consensus",
 "raw_labels":[{"labeler":"L-3f9c…","label":"kueche"},{"labeler":"L-a1b2…","label":"kueche"}],
 "model_prediction":{"label":"kueche","confidence":0.91,"model_version":"siglip-probe-2026…"},
 "suggested_split":"train"}
```

`meta.json` enthält Format-Version, App-Version, Timestamp, Seed, Ratios, Label-Set, Counts pro
Split/Label, `manifest_sha256` und die Gruppen→Split-Zuordnung. Gleicher DB-Stand + Seed ⇒
byte-identisches Manifest.

- **Split ohne Leakage:** Splits werden pro `group_id` vergeben (größenbalanciert, deterministisch);
  eine Gruppe landet nie in zwei Splits. Mit `--previous <manifest>` (bzw. Checkbox im Admin-UI)
  bleiben Zuordnungen früherer Exporte erhalten – Gruppen wandern bei Re-Exports nie von train nach test.
- **Library-Bilder** haben als `group_id` die Szenenklasse – ein Gruppen-Split würde ganze Klassen in
  einen Split legen. Deshalb gehen sie standardmäßig **immer nach `train`** (`--library-split group`
  ändert das). Val/Test bestehen damit aus echten Wohnungsdaten.
- **Pseudonymisierung:** Labeler erscheinen als `L-<hmac>` (HMAC-SHA256 über `sub`, Salt
  `EXPORT_SALT`/`SECRET_KEY`) – keine E-Mails/Subs im Datapack. Salt stabil halten, sonst ändern sich die IDs.
- Bei wenigen Gruppen können val/test leer bleiben – `meta.json → warnings` weist darauf hin.
- Exakte Duplikate werden per sha256 entfernt; **nahezu identische Frames** (statische Kameraposition)
  nicht – sie teilen sich aber immer dieselbe Gruppe und damit denselben Split.

## Security Model

**Auth-Flow:** `/auth/login` → Provider (Authorization Code + PKCE S256, `state`, `nonce`) →
`/auth/callback`: Authlib validiert State/PKCE/Nonce und ID-Token (Signatur, `iss`, `aud`, `exp`);
`redirect_uri` kommt ausschließlich aus der Konfiguration (Allowlist, kein Wert aus dem Request);
`next` wird nur als lokaler Pfad akzeptiert (kein Open Redirect). Beim Login wird die Session komplett
neu aufgebaut (Rotation: neue `sid`, neues CSRF-Token, Vor-Login-Daten verworfen). Keine impliziten Flows.

**Session:** Starlette `SessionMiddleware` (signiertes Cookie) mit `SECRET_KEY` (≥ 32 Byte),
`HttpOnly`, `Secure` (Default), `SameSite=Lax`, absolute TTL (`SESSION_TTL_SECONDS`, 8 h).
Die Session ist zustandslos: Logout löscht das Cookie, ein bereits kopiertes Cookie bleibt bis zum
TTL-Ablauf gültig (Secret rotieren = alle Sessions ungültig).

**Autorisierung:** `/admin/**` prüft serverseitig bei *jedem* Request die ENV-Allowlist
(`ADMIN_SUBS`, verifizierte `ADMIN_EMAILS`) – ein Entfernen wirkt sofort. Labels werden immer an den
authentifizierten Labeler aus der Session gebunden; Bilder werden per sha256 referenziert, nie per
Pfad oder fortlaufender ID (keine IDOR).

**CSRF:** per-Session-Token, Pflicht für alle state-ändernden POSTs (Header `X-CSRF-Token` bei fetch,
Hidden-Field bei Formularen), Vergleich zeitkonstant.

**Bildauslieferung:** nur `GET /img/<sha256>` für angemeldete Nutzer; sha256-Regex → DB-Lookup →
serverseitig aufgelöster Pfad, `resolve()` + `is_relative_to(IMAGES_DIR)`; `nosniff`, `private`-Cache.

**Header & Härtung:** strikte CSP (`default-src 'none'`, `script-src 'self'`, kein Inline-JS/-Style,
`frame-ancestors 'none'`), `nosniff`, `Referrer-Policy: no-referrer`, COOP/CORP, HSTS (hinter TLS),
`TrustedHost` (`ALLOWED_HOSTS`), Rate-Limits (slowapi) auf Login, Submit und Lese-Endpoints,
generische Fehlermeldungen (keine Stacktraces im Client), OpenAPI/Docs nur in `ENVIRONMENT=dev`,
Logging ohne Tokens/E-Mails/Bildinhalte.

**Ingestion:** ffmpeg nur als Argumentliste (nie `shell=True`), `-protocol_whitelist file` (keine
Netz-/Playlist-Zugriffe), Typ-/Größen-/Pixel-Validierung (Pillow), Dateinamen/Gruppen sanitized,
Symlinks werden ignoriert, es werden nie URLs aus Nutzereingaben geladen (Library-Download nutzt feste
HTTPS-URLs). DB-Zugriffe ausschließlich über ORM/parametrisierte Queries.

**Threat-Model (kurz):** Angreifer ohne Account → kein Zugriff auf Bilder/API (401), Brute-Force
auf Login begrenzt, kein Secret im Repo. Angreifer mit Labeler-Account → kann nur eigene Labels
setzen, keine Admin-Funktionen, kein Path-Traversal, kein Zugriff auf Fremdbilder außer über bekannte
sha256 (256 Bit, nicht ratbar). Bösartige Dateien im Ingest → validiert, isoliert (Container
read-only, non-root, `cap_drop: ALL`). **Nicht abgedeckt:** kompromittierter OIDC-Provider/Host,
Ende-zu-Ende-Verschlüsselung der Daten auf der Platte.

**Was nie committet wird:** `.env`, `*.db`, `images/`, `images_in/`, `videos_in/`, `data/`,
`embed_cache/`, `var/`, Modelle (`*.joblib`, `*.pt`, `*.safetensors`, …), Bild-/Videodateien. Im Repo
liegen nur `.env.example` (Platzhalter) und Code; Tests erzeugen synthetische Bilder zur Laufzeit.

**Eigenen OIDC-Provider anbinden:** Client (confidential, Authorization-Code, PKCE) anlegen, exakte
Redirect-URI `https://<host>/auth/callback` registrieren, Scopes `openid email profile`, die vier
`OIDC_*`-Variablen setzen, `ADMIN_SUBS` eintragen. Der Issuer muss in Produktion `https://` sein.

**CI/CD:** `ci.yml` (ruff, ruff format, mypy, pytest+Coverage), `security.yml` (CodeQL, pip-audit,
gitleaks, Trivy fs + Image, Dependency-Review), `docker.yml` (Build → Trivy-Gate → GHCR-Push mit
Provenance/SBOM). Alle Workflows: minimale `permissions`, Actions auf Commit-SHA gepinnt,
`concurrency`. Dependabot für uv, Docker und GitHub-Actions. Dependencies sind in `uv.lock`
(mit Hashes) gepinnt.

## Lokal end-to-end testen (Docker, inkl. OIDC)

`docker-compose.dev.yml` startet die App mit Hot-Reload **und** einen lokalen OIDC-Provider
([Dex](https://dexidp.io/)) mit Test-Usern – ohne externe Accounts. **Nur für lokal, nie exponieren.**

```bash
make dev-up        # Build + Start: App http://localhost:8000, Dex http://dex.localhost:5556
make dev-demo      # synthetische Demo-Bilder/-Video nach ./images_in, ./videos_in (keine echten Daten)
make dev-ingest    # Ingest in die Dev-DB
```

Login mit `alice@example.test` / `password` (ebenso `bob@…`, `carol@…`; `admin@example.test` ist Admin).
Mit mehreren Usern in getrennten Browser-Fenstern labeln, um `needs_more`, `conflict`, Agreement und
den Export unter `/admin` zu sehen. `*.localhost` löst moderne Browser automatisch auf 127.0.0.1 auf.
Embeddings/Training brauchen Torch: `DEV_EXTRAS="--extra probe --extra siglip" make dev-up`, dann
`make dev-embed dev-train`. `make dev-down` entfernt Container **und** Dev-Volumes.

## Betrieb (Docker)

```bash
cp .env.example .env && $EDITOR .env
docker compose up -d --build                         # App auf 127.0.0.1:8000
docker compose run --rm app python -m scripts.ingest
docker compose run --rm app python -m scripts.embed
docker compose run --rm app python -m scripts.train_probe
docker compose run --rm app python -m scripts.export
```

Volumes: `images`, `db` (SQLite/WAL), `embed-cache`, `models`, `exports`, `hf-cache`;
`./images_in` und `./videos_in` werden read-only gemountet. Davor gehört ein TLS-terminierender
Reverse-Proxy (Caddy/nginx/Traefik); `FORWARDED_ALLOW_IPS` auf dessen Adresse setzen. Ein Authentik-
Block steht auskommentiert in `docker-compose.yml`. Web-only-Image ohne Torch:
`docker build --build-arg EXTRAS="" .` (Embed/Train dann lokal ausführen).

SQLite: WAL, `busy_timeout`, Schreibzugriffe als kurze `BEGIN IMMEDIATE`-Transaktionen (serialisiert,
keine Lock-Upgrades), Ingest committet in Batches. Ausgelegt für wenige gleichzeitige Labeler und
**einen** Uvicorn-Worker (Rate-Limit-Zähler liegen im Speicher).

## GitHub-Repo-Einstellungen (manuell)

Code kann das nicht selbst setzen – bitte in *Settings → Branches* eine Branch-Protection-Regel
für `main` anlegen: Pull Request + Review erforderlich, **Required status checks** (`CI / lint · types · tests`,
die Jobs `CodeQL (python)`, `pip-audit`, `gitleaks`, `trivy (filesystem)`, `trivy (container image)`, `dependency review`), Branch up-to-date, keine
Force-Pushes. Zusätzlich unter *Security*: Secret Scanning + Push Protection, Dependabot Alerts und
Private Vulnerability Reporting aktivieren.

## Entwicklung

```bash
uv run pre-commit install      # ruff, ruff-format, gitleaks, large-file/EOF-Checks lokal
make check                     # lint + mypy + pytest (Coverage-Gate 90 %)
make audit                     # pip-audit
```

Siehe [CONTRIBUTING.md](CONTRIBUTING.md) und [SECURITY.md](SECURITY.md). Lizenz: [MIT](LICENSE).
