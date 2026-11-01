import { useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  BookOpen,
  Brain,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  FileText,
  FolderOpen,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  Plus,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react';

const api = async (url, options) => {
  const r = await fetch(url, options);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || 'Something went wrong.');
  return data;
};
const shortDate = (value) =>
  value
    ? new Date(value).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      })
    : '';
const average = (items) =>
  items.length ? Math.round(items.reduce((sum, item) => sum + item.score, 0) / items.length) : null;
const evidenceStatus = (check) =>
  check?.status === 'supported'
    ? 'Source check: supported'
    : check?.status === 'review'
      ? 'Source check: review suggested'
      : 'Source check: not available';

function App() {
  const [state, setState] = useState({ materials: [], attempts: [], masteryByMaterial: {} });
  const [activeTab, setActiveTab] = useState('overview');
  const [quiz, setQuiz] = useState(null);
  const [quizResult, setQuizResult] = useState(null);
  const [answers, setAnswers] = useState([]);
  const [qIndex, setQIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [toast, setToast] = useState('');
  const [mobileNav, setMobileNav] = useState(false);
  const fileRef = useRef(null);
  // The server owns saved materials and learning history; refresh after persisted changes.
  const refresh = async () => setState(await api('/api/state'));
  useEffect(() => {
    refresh().catch(() => notify('Could not connect to the study workspace.'));
  }, []);
  function notify(text) {
    setToast(text);
    setTimeout(() => setToast(''), 3500);
  }
  function navigate(tab) {
    setActiveTab(tab);
    setMobileNav(false);
  }
  async function uploadPdf(file) {
    if (!file) return;
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      notify('Please choose a PDF file.');
      return;
    }
    const fd = new FormData();
    fd.append('file', file);
    setUploading(true);
    try {
      await api('/api/materials', { method: 'POST', body: fd });
      await refresh();
      notify('PDF added to your materials.');
    } catch (e) {
      notify(e.message);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }
  async function deleteMaterial(id) {
    try {
      await api(`/api/materials/${id}`, { method: 'DELETE' });
      await refresh();
      notify('Material removed.');
    } catch (e) {
      notify(e.message);
    }
  }
  async function startQuiz(material) {
    if (!material?.id) {
      navigate('takeQuiz');
      return;
    }
    // Keep an in-progress quiz in the UI; the server saves it only after submission.
    setLoading(true);
    setQuiz(null);
    setQuizResult(null);
    setAnswers([]);
    setQIndex(0);
    try {
      const q = await api('/api/quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ materialId: material.id }),
      });
      setQuiz(q);
      navigate('quiz');
      if (q.demo)
        notify('Demo questions are shown. Add a Gemini API key for quizzes based on your PDF.');
    } catch (e) {
      notify(e.message);
    } finally {
      setLoading(false);
    }
  }
  async function submitQuiz() {
    if (answers.some((answer) => answer == null)) {
      notify('Choose an answer for every question first.');
      return;
    }
    setLoading(true);
    try {
      const result = await api('/api/quiz/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          materialId: quiz.materialId,
          title: quiz.title,
          questions: quiz.questions,
          answers,
        }),
      });
      setQuizResult(result);
      await refresh();
    } catch (e) {
      notify(e.message);
    } finally {
      setLoading(false);
    }
  }
  const attempts = state.attempts || [];
  const topics = Object.values(state.masteryByMaterial || {}).flatMap((map) =>
    Object.entries(map || {}).map(([name, value]) => [name, value])
  );
  const latest = attempts[0];
  const label =
    {
      overview: 'Overview',
      materials: 'My materials',
      takeQuiz: 'Take a quiz',
      progress: 'My progress',
      quiz: 'Quiz',
    }[activeTab] || 'Overview';

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? 'nav-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark">
            <Brain size={21} strokeWidth={2.1} />
            <span />
          </div>
          <span className="brand-name">
            echo<span>pilot</span>
          </span>
          <button
            className="icon-btn mobile-close"
            onClick={() => setMobileNav(false)}
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>
        <div className="nav-label">YOUR SPACE</div>
        <nav className="main-nav">
          <button
            className={`nav-item ${activeTab === 'overview' ? 'selected' : ''}`}
            onClick={() => navigate('overview')}
          >
            <LayoutDashboard size={18} /> Overview
          </button>
          <button
            className={`nav-item ${activeTab === 'materials' ? 'selected' : ''}`}
            onClick={() => navigate('materials')}
          >
            <FolderOpen size={18} /> My materials{' '}
            <span className="nav-count">{state.materials.length}</span>
          </button>
          <button
            className={`nav-item ${activeTab === 'takeQuiz' || activeTab === 'quiz' ? 'selected' : ''}`}
            onClick={() => navigate('takeQuiz')}
          >
            <CircleHelp size={18} /> Take a quiz
          </button>
          <button
            className={`nav-item ${activeTab === 'progress' ? 'selected' : ''}`}
            onClick={() => navigate('progress')}
          >
            <Brain size={18} /> My progress
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="profile-avatar">H</div>
          <div className="profile-copy">
            <strong>Hey, learner</strong>
            <small>Your study space</small>
          </div>
        </div>
      </aside>
      {mobileNav && (
        <button className="nav-scrim" onClick={() => setMobileNav(false)} aria-label="Close menu" />
      )}
      <main className="main-area">
        <header className="topbar">
          <button
            className="icon-btn mobile-menu"
            onClick={() => setMobileNav(true)}
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            Your space <ChevronRight size={14} /> <strong>{label}</strong>
          </div>
          <div className="topbar-actions">
            <button className="avatar-small" aria-label="Learner">
              H
            </button>
          </div>
        </header>
        <div className="page-content">
          {activeTab === 'overview' && (
            <Overview
              state={state}
              latest={latest}
              topicCount={topics.length}
              onUpload={() => fileRef.current?.click()}
              onTakeQuiz={() => navigate('takeQuiz')}
              onTab={navigate}
            />
          )}
          {activeTab === 'materials' && (
            <Materials
              state={state}
              onUpload={() => fileRef.current?.click()}
              onDelete={deleteMaterial}
              onTakeQuiz={startQuiz}
              uploading={uploading}
              loading={loading}
            />
          )}
          {activeTab === 'takeQuiz' && (
            <SelectMaterial
              state={state}
              onUpload={() => fileRef.current?.click()}
              onStart={startQuiz}
              uploading={uploading}
              loading={loading}
            />
          )}
          {activeTab === 'progress' && <Progress state={state} />}
          {activeTab === 'quiz' && (
            <Quiz
              quiz={quiz}
              result={quizResult}
              answers={answers}
              setAnswers={setAnswers}
              index={qIndex}
              setIndex={setQIndex}
              onStart={() => navigate('takeQuiz')}
              onSubmit={submitQuiz}
              loading={loading}
              onBack={() => navigate('takeQuiz')}
            />
          )}
        </div>
      </main>
      <input
        ref={fileRef}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={(e) => uploadPdf(e.target.files?.[0])}
      />
      {toast && (
        <div className="toast">
          <Sparkles size={16} />
          {toast}
        </div>
      )}
    </div>
  );
}

function Overview({ state, latest, topicCount, onUpload, onTakeQuiz, onTab }) {
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
  const count = state.materials.length;
  const completed = state.attempts.length;
  const avg = average(state.attempts);
  return (
    <div className="overview-page">
      <div className="welcome-row">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-dot" /> YOUR LEARNING SPACE
          </div>
          <h1>
            {greeting}, learner <span className="wave">✳</span>
          </h1>
          <p className="welcome-sub">A little progress every day adds up. Ready when you are.</p>
        </div>
        <button className="button button-primary" onClick={onTakeQuiz}>
          <CircleHelp size={16} /> Take a quiz
        </button>
      </div>
      <div className="hero-card">
        <div className="hero-content">
          <div className="hero-chip">
            <Sparkles size={13} /> YOUR PERSONAL STUDY COMPANION
          </div>
          <h2>
            Make your study time
            <br />
            feel <em>more like you.</em>
          </h2>
          <p>
            Choose a study material, practice what matters, and see your understanding grow — one
            small step at a time.
          </p>
          <button className="hero-cta" onClick={onUpload}>
            <Upload size={16} /> Add a study material <ArrowRight size={15} />
          </button>
        </div>
        <div className="hero-art">
          <div className="art-orbit orbit-a" />
          <div className="art-orbit orbit-b" />
          <div className="art-center">
            <Brain size={47} />
          </div>
          <div className="art-float float-top">
            <Sparkles size={15} />
            <span>curiosity</span>
          </div>
          <div className="art-float float-bottom">
            <CheckCircle2 size={16} />
            <span>growing every day</span>
          </div>
          <div className="art-dot dot-one" />
          <div className="art-dot dot-two" />
        </div>
      </div>
      <div className="stats-grid">
        <Stat
          icon={<FileText size={18} />}
          label="Study materials"
          value={count}
          note={
            count
              ? `${count} PDF${count === 1 ? '' : 's'} ready to explore`
              : 'Your notes, all in one place'
          }
          tone="mint"
        />
        <Stat
          icon={<CircleHelp size={18} />}
          label="Quizzes completed"
          value={completed}
          note={completed ? 'Nice work showing up' : 'A good place to start'}
          tone="lavender"
        />
        <Stat
          icon={<Brain size={18} />}
          label="Topics in progress"
          value={topicCount}
          note={topicCount ? 'Learning at your pace' : 'Your map starts here'}
          tone="peach"
        />
        <Stat
          icon={<Sparkles size={18} />}
          label="Quiz average"
          value={avg == null ? '—' : `${avg}%`}
          note={latest ? 'Across your study sessions' : 'Take a quiz to find out'}
          tone="blue"
        />
      </div>
      <div className="overview-columns">
        <section className="panel study-panel">
          <div className="panel-head">
            <div>
              <div className="section-kicker">PICK UP WHERE YOU ARE</div>
              <h3>Your next step</h3>
            </div>
            <span className="gentle-pill">
              <Sparkles size={13} /> at your pace
            </span>
          </div>
          <div className="next-step">
            <div className="step-icon">
              <BookOpen size={21} />
            </div>
            <div className="step-copy">
              <strong>
                {count ? 'Choose a material and keep learning' : 'Start with what you’re learning'}
              </strong>
              <span>
                {count
                  ? `${count} individual material${count === 1 ? '' : 's'} ready for a focused quiz.`
                  : 'Upload a class PDF to create your first material-based quiz.'}
              </span>
            </div>
            <button className="circle-arrow" onClick={onTakeQuiz} aria-label="Choose a material">
              <ArrowRight size={17} />
            </button>
          </div>
          <div className="quick-actions">
            <button className="quick-action" onClick={onUpload}>
              <span className="qa-icon qa-mint">
                <Upload size={17} />
              </span>
              <span>
                <strong>Upload notes</strong>
                <small>PDF files</small>
              </span>
              <ChevronRight size={16} />
            </button>
            <button className="quick-action" onClick={onTakeQuiz}>
              <span className="qa-icon qa-lilac">
                <CircleHelp size={17} />
              </span>
              <span>
                <strong>Choose a quiz</strong>
                <small>Select a material</small>
              </span>
              <ChevronRight size={16} />
            </button>
          </div>
        </section>
        <section className="panel topics-panel">
          <div className="panel-head">
            <div>
              <div className="section-kicker">YOUR MATERIALS</div>
              <h3>Ready to practice</h3>
            </div>
            <button className="text-link" onClick={() => onTab('materials')}>
              See all <ArrowRight size={14} />
            </button>
          </div>
          {state.materials.length ? (
            <div className="overview-material-list">
              {state.materials.slice(0, 3).map((m) => (
                <div className="overview-material" key={m.id}>
                  <FileText size={17} />
                  <span title={m.name}>{m.name}</span>
                  <button onClick={() => onTakeQuiz()} aria-label={`Choose a quiz for ${m.name}`}>
                    <ArrowRight size={15} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="topics-empty">
              <div className="topic-spark">
                <Upload size={18} />
              </div>
              <p>Your materials will show up here after you upload a PDF.</p>
              <small>Each file stays separate for focused practice.</small>
            </div>
          )}
        </section>
      </div>
      <section className="recent-section">
        <div className="recent-head">
          <div>
            <div className="section-kicker">YOUR STUDY HISTORY</div>
            <h3>Recent quizzes</h3>
          </div>
          <button className="text-link" onClick={() => onTab('progress')}>
            View progress <ArrowRight size={14} />
          </button>
        </div>
        {state.attempts.length ? (
          <div className="recent-grid">
            {state.attempts.slice(0, 3).map((a) => (
              <div className="recent-card" key={a.id}>
                <div className="recent-icon">
                  <CircleHelp size={17} />
                </div>
                <div>
                  <strong>{a.materialName || 'Study material'}</strong>
                  <small>
                    {a.title || 'Quiz'} · {a.score}% · {shortDate(a.createdAt)}
                  </small>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="recent-empty">
            <CircleHelp size={18} />
            <span>Your quiz history will appear here.</span>
            <button onClick={onTakeQuiz}>
              Choose a material <ArrowRight size={14} />
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

function Stat({ icon, label, value, note, tone }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${tone}`}>{icon}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </div>
  );
}

function Materials({ state, onUpload, onDelete, onTakeQuiz, uploading, loading }) {
  return (
    <div className="subpage">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-dot" /> YOUR LIBRARY
          </div>
          <h1>My materials</h1>
          <p>Every PDF is kept separate, with its own quiz history and learning progress.</p>
        </div>
        <button className="button button-primary" onClick={onUpload} disabled={uploading}>
          {uploading ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />} Add a PDF
        </button>
      </div>
      <UploadBanner onClick={onUpload} />
      <div className="library-head">
        <h3>
          {state.materials.length ? `Your library · ${state.materials.length}` : 'Your library'}
        </h3>
        <span>INDIVIDUAL PDFS</span>
      </div>
      {state.materials.length ? (
        <div className="material-list">
          {state.materials.map((m) => (
            <article className="material-card" key={m.id}>
              <div className="pdf-icon">
                <FileText size={21} />
              </div>
              <div className="material-main">
                <strong title={m.name}>{m.name}</strong>
                <span>
                  {m.pages ? `${m.pages} pages` : 'PDF'} · {Math.max(1, Math.round(m.size / 1024))}{' '}
                  KB · Added {shortDate(m.addedAt)}
                </span>
              </div>
              <button
                className="button button-secondary material-quiz-button"
                disabled={loading}
                onClick={() => onTakeQuiz(m)}
              >
                <CircleHelp size={15} /> Take quiz
              </button>
              <button
                className="icon-btn delete-material"
                title={`Remove ${m.name}`}
                onClick={() => onDelete(m.id)}
              >
                <Trash2 size={16} />
              </button>
            </article>
          ))}
        </div>
      ) : (
        <EmptyMaterials onUpload={onUpload} />
      )}
    </div>
  );
}

function UploadBanner({ onClick }) {
  return (
    <button className="upload-banner" onClick={onClick}>
      <div className="upload-banner-icon">
        <Upload size={21} />
      </div>
      <div>
        <strong>Add a study material</strong>
        <span>
          Choose a text-based PDF, up to 20 MB. Quizzes are generated from one selected file at a
          time.
        </span>
      </div>
      <ArrowRight size={18} />
    </button>
  );
}
function EmptyMaterials({ onUpload }) {
  return (
    <div className="empty-library">
      <div className="empty-illustration">
        <BookOpen size={28} />
        <Sparkles size={17} />
      </div>
      <h3>Your library is ready for a first chapter</h3>
      <p>Upload a class note or reading to create your first focused quiz.</p>
      <button className="button button-secondary" onClick={onUpload}>
        <Upload size={16} /> Choose a PDF
      </button>
    </div>
  );
}

function SelectMaterial({ state, onUpload, onStart, uploading, loading }) {
  return (
    <div className="subpage">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-dot" /> MATERIAL-BASED PRACTICE
          </div>
          <h1>Take a quiz</h1>
          <p>Choose one study material. Your questions and progress will stay with that file.</p>
        </div>
      </div>
      {state.materials.length ? (
        <>
          <div className="quiz-select-grid">
            {state.materials.map((material) => {
              const history = state.attempts.filter((a) => a.materialId === material.id);
              const avg = average(history);
              return (
                <article className="quiz-material-card" key={material.id}>
                  <div className="quiz-material-icon">
                    <FileText size={22} />
                  </div>
                  <span className="material-ready-label">
                    <Check size={13} /> READY TO STUDY
                  </span>
                  <h3 title={material.name}>{material.name}</h3>
                  <p>
                    {material.pages ? `${material.pages} pages` : 'PDF document'} · Added{' '}
                    {shortDate(material.addedAt)}
                  </p>
                  <div className="quiz-material-stats">
                    <span>
                      <strong>{history.length}</strong> {history.length === 1 ? 'quiz' : 'quizzes'}
                    </span>
                    <span>
                      <strong>{avg == null ? '—' : `${avg}%`}</strong> average
                    </span>
                  </div>
                  <button
                    className="button button-primary"
                    disabled={loading || uploading}
                    onClick={() => onStart(material)}
                  >
                    {loading ? (
                      <LoaderCircle className="spin" size={16} />
                    ) : (
                      <CircleHelp size={16} />
                    )}{' '}
                    Start quiz <ArrowRight size={15} />
                  </button>
                </article>
              );
            })}
          </div>
          <div className="adaptive-note">
            <Sparkles size={15} />
            <span>
              Each quiz uses only the PDF you choose and adapts to your previous answers for that
              material.
            </span>
          </div>
        </>
      ) : (
        <div className="quiz-no-materials">
          <EmptyMaterials onUpload={onUpload} />
        </div>
      )}
    </div>
  );
}

function Progress({ state }) {
  const byMaterial = new Map();
  for (const m of state.materials) byMaterial.set(m.id, { id: m.id, name: m.name, attempts: [] });
  for (const a of state.attempts) {
    const id = a.materialId || 'legacy';
    if (!byMaterial.has(id))
      byMaterial.set(id, { id, name: a.materialName || 'Removed material', attempts: [] });
    byMaterial.get(id).attempts.push(a);
  }
  const groups = [...byMaterial.values()].filter(
    (g) => g.attempts.length || state.materials.some((m) => m.id === g.id)
  );
  return (
    <div className="subpage">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="eyebrow-dot" /> YOUR LEARNING MAP
          </div>
          <h1>My progress</h1>
          <p>See quiz averages and review each question for every study material.</p>
        </div>
      </div>
      {!groups.length ? (
        <div className="progress-empty-card">
          <div className="topic-spark">
            <Brain size={19} />
          </div>
          <h3>No materials to track yet</h3>
          <p>Upload a PDF and complete a quiz to start building your progress.</p>
        </div>
      ) : (
        <div className="material-progress-list">
          {groups.map((group) => (
            <MaterialProgress
              key={group.id}
              group={group}
              mastery={state.masteryByMaterial?.[group.id] || {}}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function MaterialProgress({ group, mastery }) {
  const avg = average(group.attempts);
  return (
    <section className="panel material-progress-panel">
      <div className="material-progress-heading">
        <div className="pdf-icon">
          <FileText size={20} />
        </div>
        <div className="material-progress-name">
          <div className="section-kicker">STUDY MATERIAL</div>
          <h3 title={group.name}>{group.name}</h3>
        </div>
      </div>
      <div className="material-progress-stats">
        <div>
          <span>QUIZZES</span>
          <strong>{group.attempts.length}</strong>
        </div>
        <div>
          <span>QUIZ AVERAGE</span>
          <strong>{avg == null ? '—' : `${avg}%`}</strong>
        </div>
        <div>
          <span>TOPICS TRACKED</span>
          <strong>{Object.keys(mastery).length || '—'}</strong>
        </div>
      </div>
      {Object.keys(mastery).length > 0 && (
        <div className="material-topic-list">
          {Object.entries(mastery)
            .sort((a, b) => a[1].score - b[1].score)
            .slice(0, 3)
            .map(([topic, data]) => (
              <div className="material-topic-row" key={topic}>
                <span>{topic}</span>
                <div className="progress-track">
                  <div style={{ width: `${data.score}%` }} />
                </div>
                <strong>{data.score}%</strong>
              </div>
            ))}
        </div>
      )}
      <div className="material-history-heading">
        <h4>Quiz history</h4>
        <span>
          {group.attempts.length} {group.attempts.length === 1 ? 'QUIZ' : 'QUIZZES'}
        </span>
      </div>
      {group.attempts.length ? (
        <div className="history-list">
          {group.attempts.map((attempt, index) => (
            <details className="history-item" key={attempt.id || index}>
              <summary>
                <span className="history-calendar">{shortDate(attempt.createdAt)}</span>
                <span className="history-title">{attempt.title || 'Learning check-in'}</span>
                <span className="history-count">
                  {attempt.correct}/{attempt.total} correct
                </span>
                <strong
                  className={`history-score ${attempt.score >= 80 ? 'high' : attempt.score < 50 ? 'low' : ''}`}
                >
                  {attempt.score}%
                </strong>
              </summary>
              <div className="history-questions">
                {attempt.results?.length ? (
                  attempt.results.map((result, i) => (
                    <QuestionReview key={i} result={result} index={i} />
                  ))
                ) : (
                  <p className="legacy-history-note">
                    Question details weren’t saved for this earlier quiz.
                  </p>
                )}
              </div>
            </details>
          ))}
        </div>
      ) : (
        <div className="no-material-quizzes">No quizzes for this material yet.</div>
      )}
    </section>
  );
}

function QuestionReview({ result, index }) {
  const correctAnswer = result.options?.[result.answer];
  const selectedAnswer = result.options?.[result.selected];
  return (
    <article
      className={`history-question ${result.correct ? 'question-correct' : 'question-incorrect'}`}
    >
      <div className="history-question-top">
        <span className="question-number-small">QUESTION {String(index + 1).padStart(2, '0')}</span>
        <span className="history-answer-state">
          {result.correct ? (
            <>
              <Check size={13} /> Correct
            </>
          ) : (
            <>Review this one</>
          )}
        </span>
      </div>
      <strong className="history-question-prompt">
        {result.question || `Question text wasn’t saved for this earlier quiz · ${result.topic}`}
      </strong>
      {selectedAnswer && (
        <p>
          <span>Your answer:</span> {selectedAnswer}
        </p>
      )}
      {correctAnswer && !result.correct && (
        <p>
          <span>Correct answer:</span> {correctAnswer}
        </p>
      )}
      {result.explanation && <p className="question-explanation">{result.explanation}</p>}
      {result.evidence && (
        <p className="question-evidence">
          <span>PDF evidence{result.sourceId ? ` · ${result.sourceId}` : ''}:</span> “
          {result.evidence}”
        </p>
      )}
      <p className="source-check">{evidenceStatus(result.evidenceCheck)}</p>
    </article>
  );
}

function Quiz({
  quiz,
  result,
  answers,
  setAnswers,
  index,
  setIndex,
  onStart,
  onSubmit,
  loading,
  onBack,
}) {
  const q = quiz?.questions?.[index];
  if (!quiz)
    return (
      <div className="subpage quiz-intro-page">
        <div className="quiz-intro-card">
          <div className="quiz-intro-art">
            <div className="quiz-orb">
              <CircleHelp size={38} />
            </div>
            <span className="orbit-label ol-one">one material</span>
            <span className="orbit-label ol-two">your own pace</span>
            <span className="quiz-star qs-one">✳</span>
            <span className="quiz-star qs-two">✦</span>
          </div>
          <div className="eyebrow">
            <span className="eyebrow-dot" /> SELECT A STUDY MATERIAL
          </div>
          <h1>Choose a PDF to practice.</h1>
          <p>Each quiz uses one material and builds on your previous results for that same file.</p>
          <button className="button button-primary" onClick={onStart}>
            <FolderOpen size={16} /> Choose a material <ArrowRight size={16} />
          </button>
        </div>
        <button className="back-overview" onClick={onBack}>
          Back to materials
        </button>
      </div>
    );
  if (result)
    return (
      <div className="subpage quiz-result-page">
        <div className="result-card">
          <div className="result-confetti">✦</div>
          <div className="result-score-ring">
            <div>
              <strong>{result.attempt.score}%</strong>
              <small>YOUR SCORE</small>
            </div>
          </div>
          <div className="eyebrow">
            <span className="eyebrow-dot" /> THAT’S A WRAP · {quiz.materialName}
          </div>
          <h1>
            {result.attempt.score >= 80
              ? 'You’re finding your rhythm.'
              : result.attempt.score >= 50
                ? 'You’re building understanding.'
                : 'Every try is a good start.'}
          </h1>
          <p>
            You got{' '}
            <strong>
              {result.attempt.correct} of {result.attempt.total}
            </strong>{' '}
            right. Your progress for this material has been updated.
          </p>
          <div className="answer-review">
            {result.attempt.results.map((r, i) => (
              <div className={`review-row ${r.correct ? 'correct' : 'incorrect'}`} key={i}>
                <span className="review-status">
                  {r.correct ? <Check size={14} /> : <X size={14} />}
                </span>
                <div>
                  <strong>{r.topic}</strong>
                  <small>{r.correct ? 'Nicely done' : 'A chance to revisit'}</small>
                </div>
                <span className="review-explain">
                  {r.explanation}
                  {r.evidence && (
                    <small className="review-evidence">
                      PDF evidence{r.sourceId ? ` · ${r.sourceId}` : ''}: “{r.evidence}”
                    </small>
                  )}
                  <small className="review-evidence">{evidenceStatus(r.evidenceCheck)}</small>
                </span>
              </div>
            ))}
          </div>
          <div className="result-actions">
            <button className="button button-primary" onClick={onStart}>
              <Sparkles size={16} /> Choose another quiz
            </button>
            <button className="button button-secondary" onClick={onBack}>
              Back to materials
            </button>
          </div>
        </div>
      </div>
    );
  return (
    <div className="subpage quiz-play-page">
      <div className="quiz-play-head">
        <button className="back-link" onClick={onBack}>
          ‹ Choose another material
        </button>
        <span className="quiz-live-label">
          <FileText size={14} /> {quiz.materialName}
        </span>
        <span className="question-count">
          {index + 1} <span>/</span> {quiz.questions.length}
        </span>
      </div>
      <div className="quiz-progress-line">
        <span style={{ width: `${((index + 1) / quiz.questions.length) * 100}%` }} />
      </div>
      <div className="question-card">
        <div className="question-meta">
          <span className="topic-tag">
            <BookOpen size={13} />
            {q.topic}
          </span>
          <span className={`difficulty-tag ${q.difficulty || 'steady'}`}>
            {q.difficulty === 'gentle'
              ? 'Easy does it'
              : q.difficulty === 'stretch'
                ? 'A little stretch'
                : 'Steady pace'}
          </span>
        </div>
        <div className="question-number">QUESTION {String(index + 1).padStart(2, '0')}</div>
        <h1>{q.question}</h1>
        <div className="option-list">
          {q.options.map((opt, i) => (
            <button
              key={i}
              className={`answer-option ${answers[index] === i ? 'chosen' : ''}`}
              onClick={() =>
                setAnswers((a) => {
                  const next = [...a];
                  next[index] = i;
                  return next;
                })
              }
            >
              <span className="option-letter">{String.fromCharCode(65 + i)}</span>
              <span>{opt}</span>
              {answers[index] === i && <CheckCircle2 size={18} />}
            </button>
          ))}
        </div>
        <div className="question-footer">
          <span>
            <Sparkles size={14} /> Take your time — there’s no rush.
          </span>
          <div>
            {index > 0 && (
              <button className="button button-light" onClick={() => setIndex(index - 1)}>
                Previous
              </button>
            )}
            {index < quiz.questions.length - 1 ? (
              <button
                className="button button-primary"
                disabled={answers[index] == null}
                onClick={() => setIndex(index + 1)}
              >
                Next question <ChevronRight size={16} />
              </button>
            ) : (
              <button
                className="button button-primary"
                disabled={loading || answers[index] == null}
                onClick={onSubmit}
              >
                {loading ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />} Finish
                quiz
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
