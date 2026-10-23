#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# macOS may put its older system Ruby ahead of an installed Homebrew Ruby.
if ! ruby -e 'exit(Gem::Version.new(RUBY_VERSION) >= Gem::Version.new("3.1") ? 0 : 1)' 2>/dev/null; then
  for ruby_bin in /opt/homebrew/opt/ruby@3.1/bin /usr/local/opt/ruby@3.1/bin /opt/homebrew/opt/ruby/bin /usr/local/opt/ruby/bin; do
    if [[ -x "$ruby_bin/ruby" ]]; then
      export PATH="$ruby_bin:$PATH"
      break
    fi
  done
fi

exec bundle exec jekyll serve --livereload --host 127.0.0.1 --port 4000 "$@"
