-- modified vital_all48 by removing the between intime and intime + 2days note and the ie.subject_id in {}.
--added value to the select statement and groupby/orderby.
DROP MATERIALIZED VIEW IF EXISTS bg_all_nosummary CASCADE;
create materialized view bg_all_nosummary as
select pvt.SUBJECT_ID, pvt.HADM_ID, pvt.ICUSTAY_ID, pvt.CHARTTIME, label, valuenum,  value, valueuom--, PaO2_flag   added uom 12/18/18

from
( -- begin query that extracts the data
  select ie.subject_id, ie.hadm_id, ie.icustay_id, valueuom -- added valueuom 12/18/18
  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
      , case
        when itemid = 50800 then 'SPECIMEN'
        when itemid = 50801 then 'AADO2'
        when itemid = 50802 then 'BASEEXCESS'
        when itemid = 50803 then 'BICARBONATE'
        when itemid = 50804 then 'TOTALCO2'
        when itemid = 50805 then 'CARBOXYHEMOGLOBIN'
        when itemid = 50806 then 'CHLORIDE'
        when itemid = 50808 then 'CALCIUM'
        when itemid = 50809 then 'GLUCOSE'
        when itemid = 50810 then 'HEMATOCRIT'
        when itemid = 50811 then 'HEMOGLOBIN'
        when itemid = 50812 then 'INTUBATED'
        when itemid = 50813 then 'LACTATE'
        when itemid = 50814 then 'METHEMOGLOBIN'
        when itemid = 50815 then 'O2FLOW'
        when itemid = 50816 then 'FIO2'
        when itemid = 50817 then 'SO2' -- OXYGENSATURATION
        when itemid = 50818 then 'PCO2'
        when itemid = 50819 then 'PEEP'
        when itemid = 50820 then 'PH'
        when itemid = 50821 then 'PO2'
        when itemid = 50822 then 'POTASSIUM'
        when itemid = 50823 then 'REQUIREDO2'
        when itemid = 50824 then 'SODIUM'
        when itemid = 50825 then 'TEMPERATURE'
        when itemid = 50826 then 'TIDALVOLUME'
        when itemid = 50827 then 'VENTILATIONRATE'
        when itemid = 50828 then 'VENTILATOR'
        else null
        end as label
        , charttime
        , value
       --, case
         -- when value = 'ART' and itemid = 50800 then 'arterial'
         -- else null
          --end as PaO2_flag --added in 08/17/2018
        -- add in some sanity checks on the values
        , case
          when valuenum <= 0 then null
          when itemid = 50810 and valuenum > 100 then null -- hematocrit
          when itemid = 50816 and valuenum > 100 then null -- FiO2
          when itemid = 50817 and valuenum > 100 then null -- O2 sat
          when itemid = 50815 and valuenum >  70 then null -- O2 flow
          when itemid = 50821 and valuenum > 800 then null -- PO2
           -- conservative upper limit
        else valuenum
        end as valuenum

    from mimiciii.icustays ie
    left join mimiciii.labevents le
      on le.subject_id = ie.subject_id and le.hadm_id = ie.hadm_id
      --and le.charttime between (ie.intime - interval '6' hour) and (ie.intime + interval '2' day)
  	  --and ie.subject_id in {} --!!THIS IS THE code to get for only specific pt. 
      and le.ITEMID in
      -- blood gases
      (
        50800, 50801, 50802, 50803, 50804, 50805, 50806, 50807, 50808, 50809
        , 50810, 50811, 50812, 50813, 50814, 50815, 50816, 50817, 50818, 50819
        , 50820, 50821, 50822, 50823, 50824, 50825, 50826, 50827, 50828
        , 51545
      )
) pvt
group by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME, label, valuenum, value, valueuom --, PaO2_flag
order by pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.CHARTTIME, label, valuenum, value, valueuom --, PaO2_flag
; 

--my annotation:
--the cases in the first from under the select statement are being annotated on the icustay left joined to labevents. 