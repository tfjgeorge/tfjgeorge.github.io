---
layout: page
permalink: /software/
title: software
description: Pieces of code I developed during my research and that I am releasing open-source. For a complete list go to my <a href='https://github.com/tfjgeorge/'>github page</a>.
years: [2021]
nav: true
nav_order: 2
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f software -q @*[year={{y}}]* %}
{% endfor %}

</div>
