---
layout: default
title: Old notes
nav_exclude: true
---
## Old notes

These notes were written during the courses deep learning by Yoshua Bengio, and computer vision by Roland Memisevitch.

### Notes for course ift6268 computer vision
<ul>
  {% for post in site.categories.ift6268 %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date_to_string }})
    </li>
  {% endfor %}
</ul>

### Notes for course ift6266 deep learning
<ul>
  {% for post in site.categories.ift6266 %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date_to_string }})
    </li>
  {% endfor %}
</ul>
