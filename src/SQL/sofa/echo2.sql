DROP MATERIALIZED VIEW IF EXISTS echo2 CASCADE;
CREATE MATERIALIZED VIEW 
echo2 as(
  select ie.icustay_id, avg(weight * 0.45359237) as weight
  from mimiciii.icustays ie
  left join echodata echo
    on ie.hadm_id = echo.hadm_id
    and echo.charttime > ie.intime - interval '7' day
    and echo.charttime < ie.intime + interval '1' day
  group by ie.icustay_id
)