DROP MATERIALIZED VIEW IF EXISTS wt CASCADE;
CREATE MATERIALIZED VIEW 
wt AS
(
  SELECT ie.icustay_id
    -- ensure weight is measured in kg
    , 
	ceiling((extract( epoch from c.charttime - ie.intime))/60/60/24) as day,
	avg(CASE
        WHEN itemid IN (762, 763, 3723, 3580, 226512)
          THEN valuenum
        -- convert lbs to kgs
        WHEN itemid IN (3581)
          THEN valuenum * 0.45359237
        WHEN itemid IN (3582)
          THEN valuenum * 0.0283495231
        ELSE null
      END) AS weight

  from icu_18 ie
  left join mimiciii.chartevents c
    on ie.icustay_id = c.icustay_id
	and c.charttime >= ie.intime
  WHERE valuenum IS NOT NULL
  AND itemid IN
  (
    762, 763, 3723, 3580,                     -- Weight Kg
    3581,                                     -- Weight lb
    3582,                                     -- Weight oz
    226512 -- Metavision: Admission Weight (Kg)
  )
  AND valuenum != 0
  -- and charttime between ie.intime - interval '1' day and ie.intime + interval '1' day
  -- exclude rows marked as error
  AND c.error IS DISTINCT FROM 1
  group by ie.icustay_id,day
)