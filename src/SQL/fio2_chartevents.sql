DROP MATERIALIZED VIEW IF EXISTS fio2_chartevents CASCADE;
create materialized view fio2_chartevents as

  select SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME
    -- pre-process the FiO2s to ensure they are between 21-100%
    , max(
        case
          when itemid = 223835
            then case
              when valuenum > 0 and valuenum <= 1
                then valuenum * 100
              -- improperly input data - looks like O2 flow in litres
              when valuenum > 1 and valuenum < 21
                then null
              when valuenum >= 21 and valuenum <= 100
                then valuenum
              else null end -- unphysiological
        when itemid in (3420, 3422)
        -- all these values are well formatted
            then valuenum
        when itemid = 190 and valuenum > 0.20 and valuenum < 1
        -- well formatted but not in %
            then valuenum * 100
      else null end
    ) as valuenum
  from mimiciii.CHARTEVENTS
  where ITEMID in
  (
    3420 -- FiO2
  , 190 -- FiO2 set
  , 223835 -- Inspired O2 Fraction (FiO2)
  , 3422 -- FiO2 [measured]
  )
  -- exclude rows marked as error
  and error IS DISTINCT FROM 1
  group by SUBJECT_ID, HADM_ID, ICUSTAY_ID, CHARTTIME 
  