---
layout: page
permalink: /publications/
title: publications
description: publications and pre-prints, also see my <a href='https://scholar.google.com/citations?user=pc3_ujYAAAAJ'>google scholar</a> profile
years: [2022, 2021, 2020, 2018]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f {{ site.scholar.bibliography }} -q @*[year={{y}}]* %}
{% endfor %}

</div>
