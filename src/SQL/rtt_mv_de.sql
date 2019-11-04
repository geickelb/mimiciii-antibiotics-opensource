 select ie.icustay_id, tt.charttime
    , 1 as RRT
  from mimiciii.icustays ie
  inner join mimiciii.datetimeevents tt
    on ie.icustay_id = tt.icustay_id
    --and tt.charttime between ie.intime and ie.intime + interval '1' day
    and itemid in
    (
      -- TODO: unsure how to handle "Last dialysis"
      --  225128 -- | Last dialysis                                     | Adm History/FHPA        | datetimeevents     | Date time
        225318 -- | Dialysis Catheter Cap Change                      | Access Lines - Invasive | datetimeevents     | Date time
      , 225319 -- | Dialysis Catheter Change over Wire Date           | Access Lines - Invasive | datetimeevents     | Date time
      , 225321 -- | Dialysis Catheter Dressing Change                 | Access Lines - Invasive | datetimeevents     | Date time
      , 225322 -- | Dialysis Catheter Insertion Date                  | Access Lines - Invasive | datetimeevents     | Date time
      , 225324 -- | Dialysis CatheterTubing Change                    | Access Lines - Invasive | datetimeevents     | Date time
    )