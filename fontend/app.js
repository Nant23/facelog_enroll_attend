const API = "http://127.0.0.1:8000"

//  Page switching 
function switchPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'))
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'))
  document.getElementById(`page-${page}`).classList.add('active')
  const idx = { attendance: 0, enrollment: 1, manage: 2 }[page] ?? 0
  document.querySelectorAll('.nav-tab')[idx].classList.add('active')
  if (page === 'manage') loadStudents()
}

//  Toast 
function toast(msg, type = 'info') {
  const el = document.createElement('div')
  el.className = `toast ${type}`
  el.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ'}</span>${msg}`
  document.getElementById('toastContainer').appendChild(el)
  setTimeout(() => el.remove(), 4000)
}

// ATTENDANCE
let attendanceRunning = false
const attendanceVideo = document.getElementById('attendanceVideo')
const logEl = document.getElementById('attendanceLog')
const markedSet = new Set()

// Track spoof alerts so we don't spam them every 1.5 s
let lastSpoofAlert = 0

navigator.mediaDevices.getUserMedia({ video: true })
  .then(s => { attendanceVideo.srcObject = s })
  .catch(() => toast('Camera access denied', 'error'))

async function loadSubjects() {
  try {
    const res = await fetch(`${API}/subjects`)
    const subjects = await res.json()
    const sel = document.getElementById('subjectSelect')
    sel.innerHTML = '<option value="">Select subject</option>'
    subjects.forEach(s => {
      const o = document.createElement('option')
      o.value = s.id; o.text = s.name
      sel.appendChild(o)
    })
    if (subjects.length) loadClasses()
  } catch {
    toast('Cannot reach backend. Is the server running?', 'error')
  }
}

async function loadClasses() {
  const subjectId = document.getElementById('subjectSelect').value
  if (!subjectId) return
  try {
    const res = await fetch(`${API}/classes/${subjectId}`)
    const classes = await res.json()
    const sel = document.getElementById('classSelect')
    sel.innerHTML = '<option value="">Select class session</option>'
    classes.forEach(c => {
      const o = document.createElement('option')
      o.value = c.id; o.text = c.display
      sel.appendChild(o)
    })
  } catch {}
}

function startAttendance() {
  const classId = document.getElementById('classSelect').value
  if (!classId) { toast('Please select a class session first', 'error'); return }
  attendanceRunning = true
  document.getElementById('scanLine').classList.add('active')
  document.getElementById('statusBadge').classList.add('scanning')
  document.getElementById('statusText').textContent = 'Scanning for faces…'
  captureLoop()
  toast('Attendance started', 'success')
}

function stopAttendance() {
  attendanceRunning = false
  document.getElementById('scanLine').classList.remove('active')
  document.getElementById('statusBadge').classList.remove('scanning')
  document.getElementById('statusText').textContent = `Stopped — ${markedSet.size} marked`
  toast('Attendance stopped', 'info')
}

async function captureLoop() {
  if (!attendanceRunning) return

  const canvas = document.createElement('canvas')
  canvas.width = attendanceVideo.videoWidth
  canvas.height = attendanceVideo.videoHeight
  canvas.getContext('2d').drawImage(attendanceVideo, 0, 0)

  canvas.toBlob(async blob => {
    try {
      const classId = document.getElementById('classSelect').value
      const form = new FormData()
      form.append('file', blob)
      form.append('class_id', classId)

      const res = await fetch(`${API}/recognize`, { method: 'POST', body: form })
      const data = await res.json()
      const results = data.results ?? []  // new array format

      let spoofFound = false

      for (const face of results) {
        if (face.name === 'Spoof') {
          spoofFound = true

        } else if (face.name && face.name !== 'Unknown' && face.uid) {
          if (!markedSet.has(face.uid)) {
            markedSet.add(face.uid)
            addLogEntry(face.name, face.uid, face.time)
            document.getElementById('logCount').textContent = `(${markedSet.size})`
          }
        }
      }

      // Throttle spoof warnings to once every 5 seconds
      if (spoofFound) {
        const now = Date.now()
        if (now - lastSpoofAlert > 5000) {
          lastSpoofAlert = now
          toast('⚠️ Spoof attempt detected! Show your real face.', 'error')
          updateStatusSpoof()
        }
      } else if (results.length > 0) {
        document.getElementById('statusText').textContent = 'Scanning for faces…'
      }

    } catch {}

    setTimeout(captureLoop, 1500)
  }, 'image/jpeg', 0.85)
}

function updateStatusSpoof() {
  const badge = document.getElementById('statusBadge')
  const text = document.getElementById('statusText')
  text.textContent = '⚠️ Spoof detected!'
  badge.style.borderColor = 'rgba(247,85,85,0.5)'
  badge.style.color = 'var(--red)'
  // Revert after 4 seconds
  setTimeout(() => {
    if (attendanceRunning) {
      text.textContent = 'Scanning for faces…'
      badge.style.borderColor = ''
      badge.style.color = ''
    }
  }, 4000)
}

function addLogEntry(name, uid, time) {
  const empty = logEl.querySelector('.empty-log')
  if (empty) empty.remove()
  const initials = name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
  const entry = document.createElement('div')
  entry.className = 'log-entry'
  entry.innerHTML = `
    <div class="log-avatar">${initials}</div>
    <div class="log-info">
      <div class="log-name">${name}</div>
      <div class="log-meta">UID: ${uid}</div>
    </div>
    <div class="log-time">${time}</div>
  `
  logEl.prepend(entry)
}

// ─────────────────────────────────────────────
// ENROLLMENT
// ─────────────────────────────────────────────
let capturedPhotos = []
let captureActive = false
const enrollVideo = document.getElementById('enrollVideo')

// ── Duplicate detection state ──
let duplicateDetected = false        // true while a known face is in frame
let lastDupCheck = 0                 // timestamp of last /check-face call
const DUP_CHECK_INTERVAL = 2000     // run the check every 2 s during capture

navigator.mediaDevices.getUserMedia({ video: true })
  .then(s => { enrollVideo.srcObject = s })
  .catch(() => {})

async function loadEnrollDepts() {
  try {
    const res = await fetch(`${API}/departments`)
    if (!res.ok) throw new Error(`Server error ${res.status}`)
    const depts = await res.json()
    const sel = document.getElementById('enrollDept')
    sel.innerHTML = '<option value="">Select department…</option>'
    depts.forEach(d => {
      const o = document.createElement('option')
      // Always use the Firestore document ID — it must match
      // the departmentId field stored on group documents.
      o.value = d.id
      o.text = d.name
      sel.appendChild(o)
    })
    if (!depts.length) toast('No departments found in database', 'error')
  } catch (err) {
    toast('Failed to load departments: ' + err.message, 'error')
  }
}

async function loadEnrollGroups() {
  const deptId = document.getElementById('enrollDept').value
  if (!deptId) return

  const sel = document.getElementById('enrollGroup')
  sel.innerHTML = '<option value="">Loading groups…</option>'

  try {
    const res = await fetch(`${API}/groups/${deptId}`)
    if (!res.ok) throw new Error(`Server error ${res.status}`)
    const groups = await res.json()
    sel.innerHTML = '<option value="">Select group…</option>'
    if (!groups.length) {
      sel.innerHTML = '<option value="">No groups found for this department</option>'
      toast('No groups found for the selected department', 'error')
      return
    }
    groups.forEach(g => {
      const o = document.createElement('option')
      o.value = g.id; o.text = g.name
      sel.appendChild(o)
    })
  } catch (err) {
    sel.innerHTML = '<option value="">Failed to load groups</option>'
    toast('Failed to load groups: ' + err.message, 'error')
  }
}

// ── Show / hide the duplicate warning banner ──
function showDuplicateWarning(name, uid) {
  duplicateDetected = true

  const banner = document.getElementById('duplicateBanner')
  document.getElementById('dupName').textContent = name
  document.getElementById('dupUid').textContent = uid
  banner.classList.add('visible')

  // Disable the submit button while duplicate is present
  document.getElementById('submitBtn').disabled = true
  document.getElementById('submitBtn').classList.add('btn-disabled')
}

function hideDuplicateWarning() {
  duplicateDetected = false
  document.getElementById('duplicateBanner').classList.remove('visible')
  document.getElementById('submitBtn').disabled = false
  document.getElementById('submitBtn').classList.remove('btn-disabled')
}

// ── Periodically check a live frame for duplicates ──
async function checkFrameForDuplicate() {
  if (!captureActive) return

  const now = Date.now()
  if (now - lastDupCheck < DUP_CHECK_INTERVAL) return
  lastDupCheck = now

  const canvas = document.createElement('canvas')
  canvas.width = enrollVideo.videoWidth || 640
  canvas.height = enrollVideo.videoHeight || 480
  canvas.getContext('2d').drawImage(enrollVideo, 0, 0)

  canvas.toBlob(async blob => {
    try {
      const form = new FormData()
      form.append('file', blob)
      const res = await fetch(`${API}/check-face`, { method: 'POST', body: form })
      const data = await res.json()

      if (data.status === 'duplicate') {
        showDuplicateWarning(data.name, data.uid)
      } else {
        // Face is unknown or absent — clear any previous warning
        if (duplicateDetected) hideDuplicateWarning()
      }
    } catch {
      // Network or server error — silently ignore so capture can continue
    }
  }, 'image/jpeg', 0.85)
}

function startCapture() {
  if (capturedPhotos.length >= 20) { toast('Already captured 20 photos!', 'info'); return }
  captureActive = true
  duplicateDetected = false
  hideDuplicateWarning()
  document.getElementById('captureBtn').classList.add('loading')
  capturePhotoLoop()
}

function capturePhotoLoop() {
  if (!captureActive || capturedPhotos.length >= 20) {
    captureActive = false
    document.getElementById('captureBtn').classList.remove('loading')
    if (capturedPhotos.length >= 20) toast('20 photos captured! Ready to enroll.', 'success')
    return
  }

  // ── Duplicate check (async, non-blocking) ──
  checkFrameForDuplicate()

  // ── If a duplicate is in frame, pause photo capture but keep checking ──
  if (duplicateDetected) {
    setTimeout(capturePhotoLoop, 600)
    return
  }

  const canvas = document.createElement('canvas')
  canvas.width = enrollVideo.videoWidth || 640
  canvas.height = enrollVideo.videoHeight || 480
  canvas.getContext('2d').drawImage(enrollVideo, 0, 0)
  canvas.toBlob(blob => {
    capturedPhotos.push(blob)
    updateCaptureUI(canvas.toDataURL('image/jpeg', 0.5))
    setTimeout(capturePhotoLoop, 600)
  }, 'image/jpeg', 0.85)
}

function updateCaptureUI(dataUrl) {
  const n = capturedPhotos.length
  document.getElementById('captureCount').textContent = n
  document.getElementById('captureProgress').style.width = (n / 20 * 100) + '%'
  const strip = document.getElementById('photoStrip')
  const img = document.createElement('img')
  img.className = 'photo-thumb' + (n === 20 ? ' done' : '')
  img.src = dataUrl
  strip.appendChild(img)
  if (strip.children.length > 10) strip.removeChild(strip.children[0])
}

function resetCapture() {
  capturedPhotos = []
  captureActive = false
  duplicateDetected = false
  hideDuplicateWarning()
  document.getElementById('captureCount').textContent = '0'
  document.getElementById('captureProgress').style.width = '0%'
  document.getElementById('photoStrip').innerHTML = ''
  document.getElementById('captureBtn').classList.remove('loading')
}

async function submitEnrollment() {
  if (duplicateDetected) {
    toast('Cannot enroll — this face is already registered!', 'error')
    return
  }

  const name = document.getElementById('enrollName').value.trim()
  const email = document.getElementById('enrollEmail').value.trim()
  const password = document.getElementById('enrollPassword').value
  const dept = document.getElementById('enrollDept').value
  const group = document.getElementById('enrollGroup').value

  if (!name || !email || !password || !dept || !group) {
    toast('Please fill all student info fields', 'error'); return
  }
  if (capturedPhotos.length < 20) {
    toast(`Capture more photos: ${capturedPhotos.length}/20 done`, 'error'); return
  }

  const btn = document.getElementById('submitBtn')
  btn.classList.add('loading')

  try {
    const form = new FormData()
    form.append('name', name)
    form.append('email', email)
    form.append('password', password)
    form.append('department', dept)
    form.append('group', group)
    capturedPhotos.forEach((blob, i) => form.append('photos', blob, `photo_${i}.jpg`))

    const res = await fetch(`${API}/enroll`, { method: 'POST', body: form })
    const data = await res.json()

    if (res.ok) {
      toast(` ${name} enrolled! UID: ${data.uid}`, 'success')
      // Clear form
      document.getElementById('enrollName').value = ''
      document.getElementById('enrollEmail').value = ''
      document.getElementById('enrollPassword').value = ''
      document.getElementById('enrollDept').value = ''
      document.getElementById('enrollGroup').innerHTML = '<option value="">Select a department first</option>'
      resetCapture()
    } else {
      toast(data.detail || 'Enrollment failed', 'error')
    }
  } catch {
    toast('Cannot reach backend. Is the server running?', 'error')
  }

  btn.classList.remove('loading')
}

// ── Init ──
loadSubjects()
loadEnrollDepts()

// ─────────────────────────────────────────────
// MANAGE STUDENTS
// ─────────────────────────────────────────────
let allStudents = []
let pendingDeleteUid = null

async function loadStudents() {
  const grid = document.getElementById('studentGrid')
  const empty = document.getElementById('studentsEmpty')
  grid.innerHTML = ''
  empty.style.display = 'flex'
  empty.innerHTML = '<div style="font-size:40px;opacity:0.25">👥</div><div style="color:var(--muted);font-size:14px;margin-top:10px">Loading students…</div>'

  try {
    const res = await fetch(`${API}/students`)
    if (!res.ok) throw new Error(`Server error ${res.status}`)
    allStudents = await res.json()
    document.getElementById('statTotal').textContent = allStudents.length
    renderStudents(allStudents)
  } catch (err) {
    empty.style.display = 'flex'
    empty.innerHTML = '<div style="font-size:40px;opacity:0.25">⚠️</div><div style="color:var(--muted);font-size:14px;margin-top:10px">Failed to load. Is the server running?</div>'
    toast('Cannot load students: ' + err.message, 'error')
  }
}

function filterStudents() {
  const q = document.getElementById('studentSearch').value.toLowerCase()
  if (!q) { renderStudents(allStudents); return }
  renderStudents(allStudents.filter(s =>
    s.name.toLowerCase().includes(q) ||
    s.email.toLowerCase().includes(q) ||
    s.uid.toLowerCase().includes(q)
  ))
}

function renderStudents(students) {
  const grid = document.getElementById('studentGrid')
  const empty = document.getElementById('studentsEmpty')
  grid.innerHTML = ''

  if (!students.length) {
    empty.style.display = 'flex'
    empty.innerHTML = `<div style="font-size:40px;opacity:0.25">👥</div><div style="color:var(--muted);font-size:14px;margin-top:10px">${allStudents.length ? 'No students match your search.' : 'No students enrolled yet.'}</div>`
    return
  }

  empty.style.display = 'none'

  students.forEach(s => {
    const initials = s.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) || '?'
    const card = document.createElement('div')
    card.className = 'student-card'
    card.innerHTML = `
      <div class="student-avatar">${initials}</div>
      <div class="student-info">
        <div class="student-name">${s.name}</div>
        <div class="student-email">${s.email}</div>
        <div class="student-uid-pill">${s.uid}</div>
      </div>
      <button class="delete-btn" title="Delete student" onclick="openDeleteModal('${s.uid}','${s.name.replace(/'/g,"\\'")}')">
        🗑️
      </button>
    `
    grid.appendChild(card)
  })
}

function openDeleteModal(uid, name) {
  pendingDeleteUid = uid
  document.getElementById('deleteModalBody').innerHTML =
    `This will permanently remove <strong>${name}</strong> from Firebase Auth, Firestore, the face dataset, and retrain the recognition model.<br><br>This cannot be undone.`
  document.getElementById('deleteModalOverlay').classList.add('visible')
}

function closeDeleteModal() {
  document.getElementById('deleteModalOverlay').classList.remove('visible')
  pendingDeleteUid = null
}

async function confirmDelete() {
  if (!pendingDeleteUid) return
  const uid = pendingDeleteUid
  const btn = document.getElementById('confirmDeleteBtn')
  btn.classList.add('loading')
  btn.disabled = true

  try {
    const res = await fetch(`${API}/students/${uid}`, { method: 'DELETE' })
    const data = await res.json()

    if (res.ok) {
      toast('✅ ' + data.detail, 'success')
      closeDeleteModal()
      allStudents = allStudents.filter(s => s.uid !== uid)
      document.getElementById('statTotal').textContent = allStudents.length
      filterStudents()
    } else {
      toast(data.detail || 'Delete failed', 'error')
    }
  } catch {
    toast('Cannot reach backend. Is the server running?', 'error')
  }

  btn.classList.remove('loading')
  btn.disabled = false
}

