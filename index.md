---
layout: default
title: Home
---

I am a Phd student interested in optimization applied to (deep) neural networks, under the supervision of [Pascal Vincent](http://www.iro.umontreal.ca/~vincentp/) at [MILA](https://mila.umontreal.ca/). This website is intended at sharing some notes or short articles that I wrote during my exploration of deep learning.

For a list of publications, please refer to [scholar](https://scholar.google.fr/citations?user=pc3_ujYAAAAJ).

## Recent highlights 

 - [Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis](http://papers.nips.cc/paper/8164-fast-approximate-natural-gradient-descent-in-a-kronecker-factored-eigenbasis) - NIPS 2018 - [Poster](/assets/docs/EKFAC-NeurIPS2018.pdf)

## Notes
<ul>
  {% for post in site.categories.note %}
    {% if post.draft != true %}
      <li>
        <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date_to_string }})
      </li>
    {% endif %}
  {% endfor %}
</ul>

<a href="/page/old-notes">Older notes</a>
