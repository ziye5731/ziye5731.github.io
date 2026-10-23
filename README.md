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

Blog articles use `_posts/YYYY-MM-DD-slug.md`. Set `categories: [blog]` in
their front matter to include them in the Blog list. Existing article
permalinks are explicit so that filenames can change without changing URLs.

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
