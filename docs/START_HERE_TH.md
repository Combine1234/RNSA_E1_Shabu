# เริ่มต่อบน Codex เครื่องใหม่

เอกสารวันที่ 6 กันยายน 2026 — repo นี้มีโค้ดและเอกสารส่งต่อ ไม่ได้บรรจุข้อมูล
การแข่งขัน โมเดลที่ฝึกแล้ว หรือข้อมูลล็อกอิน งานหนักและ artifacts ยังอยู่บน Kaggle

## สถานะที่ต้องรู้ก่อน

- ส่งครั้งแรกแล้ว: submission `56035531`, notebook
  `fronktja/rsna-knee-06-submission` Version 1
- Public AUC **0.891**; อันดับ **1,826** เป็นภาพสถานะวันที่ 5 กันยายน ไม่ใช่อันดับสด
- **ยังไม่ถึงเป้าหมาย Top 100 / 0.945** แม้ notebook และการตรวจ runtime จะผ่าน
- cache ทั้งสองส่วนและโมเดลทั้งสองสายทำเสร็จแล้ว ไม่ต้องเริ่มใหม่ทั้งหมด
- โมเดลที่ส่งจริงคือ Raptor/CoAtNet ตรึง encoder แล้วฝึกหัวโมเดล 5 folds
  ใช้ slice-offset TTA 3 ตำแหน่ง; DINO ไม่ได้อยู่ในผลสุดท้ายเพราะ blend ไม่ผ่าน gate
- การอนุมัติ Submit เดิมใช้ไปแล้ว ห้ามส่งครั้งต่อไปเองโดยไม่ขออนุมัติใหม่

## วิธีเริ่ม

1. Clone repo และเปิดโฟลเดอร์นี้ใน Codex เครื่องใหม่:

   ```bash
   git clone https://github.com/Combine1234/RNSA_E1_Shabu.git
   cd RNSA_E1_Shabu
   ```

2. ให้ Codex อ่าน [AGENTS.md](../AGENTS.md),
   [คู่มือส่งต่อฉบับละเอียด](CODEX_HANDOFF.md) และ
   [บันทึกผลการทำงาน](implementation_status.md) ก่อนทำอะไรต่อ
3. ล็อกอิน Kaggle บนเครื่องใหม่ด้วยบัญชีที่เข้าถึง notebooks ของ `fronktja` ได้
   การ clone Git ไม่ได้ให้สิทธิ์เข้าถึง private Kaggle artifacts อัตโนมัติ
   ห้ามคัดลอก token หรือ cookies ลง repo
4. ตรวจสถานะและ version ของ artifacts เดิมก่อน รันได้เฉพาะ unit tests เบา ๆ
   บนเครื่องใหม่ ส่วน DICOM, feature extraction, training และ inference ใช้ Kaggle ฟรี
5. วิเคราะห์ข้อจำกัด validation ในหัวข้อ 12 ของคู่มือละเอียดก่อนวางแผนทดลองใหม่
   โดยเฉพาะการใช้ gold labels เลือก epoch และประวัติข้อมูลฝึกของ Raptor ที่ยังไม่ยืนยัน

## ข้อความที่ส่งให้ Codex เครื่องใหม่ได้เลย

> อ่าน AGENTS.md, docs/CODEX_HANDOFF.md และ docs/implementation_status.md ก่อน
> งานส่งครั้งแรกเสร็จแล้วได้ Public AUC 0.891 ซึ่งยังไม่ถึง Top 100 ตรวจ artifacts
> ที่มีอยู่บน Kaggle ก่อน ห้ามรัน cache หรือ training เดิมซ้ำโดยไม่จำเป็น ให้ตรวจ
> validation และ provenance ของ Raptor แล้วเสนอการทดลองขั้นต่อไป งานหนักใช้
> Kaggle ฟรีเท่านั้น ห้ามดาวน์โหลด DICOM, pixel cache หรือ checkpoints ลงเครื่อง
> ห้ามส่ง competition submission เพิ่มโดยไม่ได้รับอนุมัติจากผมใหม่

## สิ่งที่ต้องคงไว้

repo นี้เป็นสาธารณะตอนจัดทำ การเปลี่ยนเป็น private ภายหลังไม่ได้ลบการเปิดเผยเดิม
จึงไม่มีรายงานผู้ป่วย, study IDs จริง, row-level predictions หรือ secrets ใน Git
ดูไฟล์ JSON/CSV/log ขนาดเล็กจาก Kaggle ได้เฉพาะรายการที่เลือกชัดเจน และเก็บนอก Git
ที่ `/tmp` หรือ Portable SSD ที่อนุมัติไว้ ต้องตรวจ mount ใหม่บนเครื่องใหม่เสมอ

Heartbeat เดิมถูกพักหลังได้คะแนน และไม่ย้ายตาม Git อัตโนมัติ งาน saved run ที่ส่ง
ให้ Kaggle ไปแล้วไม่ต้องเปิดเครื่องค้าง แต่ไม่ควรสมมติว่า Codex บนเครื่องที่ปิดอยู่
จะเปิดขั้นตอนถัดไปให้เอง

คู่มือละเอียดรวมลิงก์ notebooks, versions/run IDs, inputs, โมเดล, hash ของ weights,
hyperparameters, label/fold conventions, gates, ผลทดสอบ, ข้อจำกัด และคำสั่ง deploy
แยกจากคำสั่งตรวจสถานะไว้แล้ว ไม่ต้องโอน virtual environment หรือไฟล์ชั่วคราวเดิม
