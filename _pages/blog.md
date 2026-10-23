---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: false
---

Posts on research, methods, and ideas.

{% assign blog_posts = site.posts | where_exp: "post", "post.categories contains 'blog'" %}
{% for post in blog_posts %}
- [{{ post.title }}]({{ post.url | relative_url }})
{% else %}
New posts will appear here.
{% endfor %}
