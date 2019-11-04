DROP MATERIALIZED VIEW IF EXISTS vaso_mv CASCADE;
CREATE MATERIALIZED VIEW vaso_mv AS
(select ie.icustay_id
    -- case statement determining whether the ITEMID is an instance of vasopressor usage
    , max(case when itemid = 221906 then rate end) as rate_norepinephrine
    , max(case when itemid = 221289 then rate end) as rate_epinephrine
    , max(case when itemid = 221662 then rate end) as rate_dopamine
    , max(case when itemid = 221653 then rate end) as rate_dobutamine
 ,ceiling((extract( epoch from mv.starttime - ie.intime))/60/60/24) as day
  from icu_18 ie
  inner join mimiciii.inputevents_mv mv
    on ie.icustay_id = mv.icustay_id 
  where itemid in (221906,221289,221662,221653)
  -- 'Rewritten' orders are not delivered to the patient
  and statusdescription != 'Rewritten'
 and mv.starttime > ie.intime
  group by ie.icustay_id,day
  order by ie.icustay_id,day)