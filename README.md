# Ziye's Homepage

Source for [ziye5731.github.io](https://ziye5731.github.io), built with Jekyll
and the AcademicPages theme.

## Content

- `_pages/about.md`: homepage.
- `_pages/publications.md`: publications and preprints.
- `_pages/notes.md` and `_pages/notes/`: study notes.
- `_pages/blog.md` and `_posts/`: blog index and dated articles.
- `_pages/miscellaneous.md`: music and other interests.
- `_data/navigation.yml`: top navigation.
- `images/` and `notes/`: images and downloadable notes.

Each blog article has its own folder named by its slug, for example:

```text
_posts/sieve-sgmm/
  2026-10-23-sieve-sgmm.md
  assets/
    images/       # PNG previews and SVG originals
    results/      # protocol and result summaries
```

Keep the date in the Markdown filename, rather than the folder name, so
Jekyll recognizes only the article as a post. Set `categories: [blog]` to
include the article in the Blog list. Explicit permalinks keep article URLs
stable when files are reorganized.

Use ordinary Markdown paths such as `assets/images/calibration-rates.png`
and `assets/results/protocol.json`. This keeps previews self-contained when
opening a post folder in VS Code or another Markdown reader. The
`post-bundle` layout converts these links to the front matter's `asset_path`
for the website while retaining the normal `single` page style.
`assets/posts/sieve-sgmm` is a relative directory symlink to this post's
`assets/`; Jekyll's safe build copies the assets into the generated website.
The source images and result summaries are stored only in the post folder.

Use `$...$` for inline math and `$$` on separate lines for display math.
In inline math, prefer `\lVert ...\rVert`, `\lbrace ...\rbrace`, and
`_{\ast}` to prevent Markdown from interpreting TeX delimiters as formatting.
Contents lists use ordinary heading links.

## Local preview

Use Ruby 3.1 or later with Bundler, then install the dependencies:

```sh
bundle install
```

Run the local site with:

```sh
bash run_local.sh
```

Open [127.0.0.1:4000](http://127.0.0.1:4000). Content changes reload
automatically; restart after changing `_config.yml`.

## Production build

```sh
JEKYLL_ENV=production bundle exec jekyll build --safe --strict_front_matter
```

The complete website is generated in `_site/`. This directory and build
caches are ignored by Git. The site URL in `_config.yml` is the production
GitHub Pages URL. Development utilities and unused theme demonstration
pages are excluded from the generated website.

## Research materials

The experiment design for the Sieve GMM article is recorded in
[scripts/sieve-sgmm/EXPERIMENT_PLAN.md](scripts/sieve-sgmm/EXPERIMENT_PLAN.md).
Research plans and development utilities under `scripts/` are excluded from
the generated website.

## Theme attribution

Built on [AcademicPages](https://github.com/academicpages/academicpages.github.io),
which derives from [Minimal Mistakes](https://github.com/mmistakes/minimal-mistakes).
See `LICENSE` for the retained license.
