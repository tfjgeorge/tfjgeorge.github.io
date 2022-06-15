---
layout: page
permalink: /software/
title: software
description: Pieces of code I developed during my research and that I am releasing open-source. For a complete list go to my [github page](https://github.com/tfjgeorge/).
years: [1956, 1950, 1935, 1905]
nav: true
nav_order: 2
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
