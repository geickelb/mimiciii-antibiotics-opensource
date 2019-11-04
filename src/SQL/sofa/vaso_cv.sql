DROP MATERIALIZED VIEW IF EXISTS vaso_cv CASCADE;
CREATE MATERIALIZED VIEW vaso_cv AS
(select icustay_id
    , max (rate_norepinephrine)as rate_norepinephrine

    , max(rate_epinephrine) as rate_epinephrine

    , max(rate_dopamine) as rate_dopamine
    , max( rate_dobutamine) as rate_dobutamine
	,day
	from
(select ie.icustay_id
    -- case statement determining whether the ITEMID is an instance of vasopressor usage
    ,
	ceiling((extract( epoch from cv.charttime - ie.intime))/60/60/24) as day 
  ,  
	case
            when itemid = 30047 then rate / coalesce(wt.weight,ec.weight) -- measured in mcgmin
            when itemid = 30120 then rate -- measured in mcgkgmin ** there are clear errors, perhaps actually mcgmin
            else null
          end as rate_norepinephrine

    , case
            when itemid =  30044 then rate / coalesce(wt.weight,ec.weight) -- measured in mcgmin
            when itemid in (30119,30309) then rate -- measured in mcgkgmin
            else null
          end as rate_epinephrine

    , case when itemid in (30043,30307) then rate end as rate_dopamine
    , case when itemid in (30042,30306) then rate end as rate_dobutamine

  from icu_18 ie
  inner join mimiciii.inputevents_cv cv
    on ie.icustay_id = cv.icustay_id 
  left join wt
    on ie.icustay_id = wt.icustay_id
  left join echo2 ec
    on ie.icustay_id = ec.icustay_id
  where itemid in (30047,30120,30044,30119,30309,30043,30307,30042,30306)
  and rate is not null
and cv.charttime >= ie.intime
) as pvt
  group by icustay_id, day
  order by icustay_id, day)
 
