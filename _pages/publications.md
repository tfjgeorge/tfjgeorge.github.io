---
layout: page
permalink: /publications/
title: publications
description: publications and pre-prints, also see my <a href='https://scholar.google.com/citations?user=pc3_ujYAAAAJ'>google scholar</a> profile
years: [2021, 2020, 2018]
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
