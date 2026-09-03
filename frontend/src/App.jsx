import React, { useMemo, useRef, useState } from 'react';

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:5000/api'
).replace(/\/$/, '');

const STORAGE_PATIENTS = 'dermavision_patients_v3';
const STORAGE_LANGUAGE = 'dermavision_language_v3';

const I = ({ n, s = 19 }) => {
  const p = {
    width: s,
    height: s,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.9,
    strokeLinecap: 'round',
    strokeLinejoin: 'round'
  };

  const d = {
    dash: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    ),

    plus: (
      <>
        <path d="M12 5v14M5 12h14" />
      </>
    ),

    hist: (
      <>
        <path d="M3 12a9 9 0 1 0 3-6.7" />
        <path d="M3 4v5h5" />
        <path d="M12 7v5l3 2" />
      </>
    ),

    report: (
      <>
        <path d="M6 2h9l3 3v17H6z" />
        <path d="M9 13h6M9 17h6M9 9h3" />
      </>
    ),

    chart: (
      <>
        <path d="M4 19V5M4 19h17" />
        <path d="m7 15 4-4 3 2 5-7" />
      </>
    ),

    set: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19 15l1 1-2 2-1-1a7 7 0 0 1-2 1v2h-3v-2a7 7 0 0 1-2-1l-1 1-2-2 1-1a7 7 0 0 1-1-2H5v-3h2a7 7 0 0 1 1-2L7 7l2-2 1 1a7 7 0 0 1 2-1V3h3v2a7 7 0 0 1 2 1l1-1 2 2-1 1a7 7 0 0 1 1 2h2v3h-2a7 7 0 0 1-1 2z" />
      </>
    ),

    logout: (
      <>
        <path d="M10 17l5-5-5-5M15 12H3M21 19V5a2 2 0 0 0-2-2h-5" />
      </>
    ),

    search: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>
    ),

    bell: (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" />
      </>
    ),

    upload: (
      <>
        <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
      </>
    ),

    camera: (
      <>
        <path d="M4 7h4l2-2h4l2 2h4v12H4z" />
        <circle cx="12" cy="13" r="3" />
      </>
    ),

    eye: (
      <>
        <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z" />
        <circle cx="12" cy="12" r="2.5" />
      </>
    ),

    shield: (
      <>
        <path d="M12 3 20 6v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" />
        <path d="m8.5 12 2.3 2.3 4.8-5" />
      </>
    ),

    arrow: <path d="M5 12h14M13 6l6 6-6 6" />,

    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5M12 8h.01" />
      </>
    ),

    back: <path d="m15 18-6-6 6-6" />
  };

  return <svg {...p}>{d[n] || d.info}</svg>;
};

const initial = {
  name: '',
  age: '',
  gender: '',
  location: '',
  duration: '',
  size: '',
  color: '',
  shape: '',
  symptoms: '',
  family: '',
  sun: '',
  previous: ''
};

const translations = {
  en: {
    dashboard: 'Dashboard',
    newPrediction: 'New Prediction',
    history: 'Patient History',
    reports: 'Reports',
    analytics: 'Analytics',
    settings: 'Settings',
    logout: 'Logout',
    search: 'Search patients, reports...',
    researcher: 'Researcher',
    aiService: 'AI Service',
    ready: 'Ready for analysis',
    language: 'Language',
    english: 'English',
    hindi: 'हिंदी',
    marathi: 'मराठी',

    overview: 'OVERVIEW',
    goodAfternoon: 'Good afternoon, Vaishnavi',
    monitor: 'Monitor AI-assisted skin lesion analysis from one workspace.',
    newAnalysis: 'New analysis',
    totalAnalyses: 'Total analyses',
    highRisk: 'High-risk results',
    benign: 'Benign / low-risk',
    validation: 'Validation accuracy',
    recent: 'Recent analyses',
    viewAll: 'View all',
    riskDistribution: 'Risk distribution',
    aiPipeline: 'AI pipeline',
    quickActions: 'Quick actions',
    uploadImage: 'Upload image',
    patientHistory: 'Patient history',
    generateReport: 'Generate report',
    modelAnalytics: 'Model analytics',

    welcome: 'Welcome back',
    signIn: 'Sign in to your DermaVision workspace',
    email: 'Email',
    password: 'Password',
    remember: 'Remember me',
    forgot: 'Forgot password?',
    login: 'Login',
    demo: 'Demo mode: any valid email/password opens the dashboard.',

    analysisResult: 'ANALYSIS RESULT',
    aiResult: 'AI-assisted result',
    review: 'Review prediction, image evidence and explanation.',
    report: 'Report',
    uploaded: 'Uploaded lesion image',
    patient: 'Patient',
    aiPrediction: 'AI Prediction',
    confidence: 'Confidence',
    risk: 'Risk',
    disclaimer:
      'This is a research decision-support result. Final diagnosis must be made by a qualified dermatologist.',
    preprocessing: 'Preprocessing',
    segmentation: 'Segmentation',
    gradcam: 'Grad-CAM',
    enhanced: 'Enhanced image',
    lesionMask: 'Lesion mask',
    connect: 'Connect to AI module',
    realHeatmap: 'Grad-CAM explanation is not available for this prediction.',

    records: 'PATIENT RECORDS',
    analysisHistory: 'Analysis history',
    searchHistory: 'Search by patient, Patient ID or prediction',
    allRisk: 'All risk levels',
    prediction: 'Prediction',
    date: 'Date',
    noRecords: 'No patient records yet',

    newAnalysisTitle: 'Skin lesion analysis',
    newAnalysisText:
      'Enter patient metadata and upload the lesion image used for AI analysis.',
    aiReady: '● AI service ready',
    existing: 'Existing Patient',
    newPatient: 'New Patient',
    existingHelp:
      'Enter the Patient ID to fetch the saved patient profile.',
    newHelp:
      'A unique Patient ID will be generated when the first prediction is saved.',
    patientId: 'Patient ID',
    fetchPatient: 'Fetch Patient',
    fetched: 'Patient profile fetched successfully.',
    notFound:
      'Patient ID not found. Please check the ID or create a new patient.',
    newPatientId: 'New Patient ID',
    patientProfile: 'Patient & clinical metadata',
    metadataNote:
      'Metadata is stored with the patient record. The current image model predicts from the lesion image.',
    lesionImage: 'Lesion image',
    change: 'Change',
    remove: 'Remove',
    uploadSkin: 'Upload skin lesion image',
    drag: 'Drag & drop or click to browse',
    jpg: 'JPG, JPEG, PNG • Maximum 5 MB',
    patientLinked: 'Patient-linked',
    readyAnalyze: 'Ready to analyze?',
    complete: 'Complete required fields and upload an image.',
    allInputs: 'All required inputs are available.',
    analyzing: 'Analyzing...',
    analyzeAI: 'Analyze with AI',
    back: 'Back',

    patientName: 'Patient Name *',
    age: 'Age *',
    gender: 'Gender *',
    bodyLocation: 'Body Location *',
    duration: 'Lesion Duration',
    size: 'Lesion Size',
    colour: 'Lesion Colour',
    shape: 'Lesion Shape',
    symptoms: 'Symptoms',
    family: 'Family History',
    sun: 'Sun Exposure',
    previous: 'Previous Skin Cancer',
    select: 'Select',
    female: 'Female',
    male: 'Male',
    other: 'Other',
    face: 'Face',
    scalp: 'Scalp',
    neck: 'Neck',
    chest: 'Chest',
    backBody: 'Back',
    arm: 'Arm',
    leg: 'Leg',
    hand: 'Hand',
    foot: 'Foot',
    itching: 'Itching, bleeding, pain...',

    low: 'Low',
    medium: 'Medium',
    high: 'High',
    suspicious: 'Suspicious class',
    nonSuspicious: 'Non-suspicious class',
    model: 'Model',
    device: 'Device',
    probabilities: 'Class probabilities',

    reportCenter: 'REPORT CENTER',
    clinicalReport: 'Clinical-style report',
    print: 'Print / Save PDF',
    pending: 'Pending',
    clinicalDisclaimer: 'Clinical disclaimer',
    reportText:
      'This system is intended for research and decision support. It does not replace clinical examination, dermoscopy, biopsy, or diagnosis by a qualified dermatologist.',

    modelMonitoring: 'MODEL MONITORING',
    analyticsText:
      'Use evaluation metrics from the trained model instead of demo placeholders.',
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1 Score',
    currentBackend: 'Current backend',
    apiBase: 'API base URL',
    predictionEndpoint: 'Prediction endpoint',
    live: 'Live AI API',
    privacy: 'Privacy',
    privacyText:
      'Use authorized patient data only. Store sensitive information securely in a production deployment.',

    languageHint: 'Choose a language comfortable for you.',
    invalidImage: 'Please select an image file.',
    max5mb: 'Maximum file size is 5 MB.',
    unknown: 'Unknown',
    liveSessionRecords: 'Live session records',
    noRecordsYet: 'No records yet',
    currentTestAccuracy: 'Current model test accuracy',
    weightedPrecision: 'Weighted precision',
    weightedRecall: 'Weighted recall',
    weightedF1: 'Weighted F1',
    lesionAnalysis: 'Lesion analysis',
    explainability: 'Explainability',
    clinicalReportStep: 'Clinical report',
    pipeline1: 'Resize • normalize • artifact handling',
    pipeline2: 'Classification + segmentation',
    pipeline3: 'Grad-CAM / attention',
    pipeline4: 'Prediction + patient record',
    skinLesionReport: 'Skin Lesion Analysis Report',
    researchReport: 'Research decision-support report'
  },

  hi: {
    dashboard: 'डैशबोर्ड',
    newPrediction: 'नई भविष्यवाणी',
    history: 'मरीज़ इतिहास',
    reports: 'रिपोर्ट',
    analytics: 'विश्लेषण',
    settings: 'सेटिंग्स',
    logout: 'लॉगआउट',
    search: 'मरीज़ और रिपोर्ट खोजें...',
    researcher: 'रिसर्चर',
    aiService: 'AI सेवा',
    ready: 'विश्लेषण के लिए तैयार',
    language: 'भाषा',
    english: 'English',
    hindi: 'हिंदी',
    marathi: 'मराठी',

    overview: 'सारांश',
    goodAfternoon: 'नमस्ते, Vaishnavi',
    monitor: 'एक ही जगह से AI त्वचा घाव विश्लेषण देखें।',
    newAnalysis: 'नया विश्लेषण',
    totalAnalyses: 'कुल विश्लेषण',
    highRisk: 'उच्च जोखिम परिणाम',
    benign: 'सौम्य / कम जोखिम',
    validation: 'मान्यता सटीकता',
    recent: 'हाल के विश्लेषण',
    viewAll: 'सभी देखें',
    riskDistribution: 'जोखिम वितरण',
    aiPipeline: 'AI प्रक्रिया',
    quickActions: 'त्वरित कार्य',
    uploadImage: 'चित्र अपलोड करें',
    patientHistory: 'मरीज़ इतिहास',
    generateReport: 'रिपोर्ट बनाएं',
    modelAnalytics: 'मॉडल विश्लेषण',

    welcome: 'स्वागत है',
    signIn: 'अपने DermaVision कार्यक्षेत्र में साइन इन करें',
    email: 'ईमेल',
    password: 'पासवर्ड',
    remember: 'मुझे याद रखें',
    forgot: 'पासवर्ड भूल गए?',
    login: 'लॉगिन',
    demo: 'डेमो मोड: कोई भी मान्य ईमेल/पासवर्ड डैशबोर्ड खोलता है।',

    analysisResult: 'विश्लेषण परिणाम',
    aiResult: 'AI-सहायित परिणाम',
    review: 'भविष्यवाणी, चित्र और स्पष्टीकरण देखें।',
    report: 'रिपोर्ट',
    uploaded: 'अपलोड किया गया घाव चित्र',
    patient: 'मरीज़',
    aiPrediction: 'AI भविष्यवाणी',
    confidence: 'विश्वास स्तर',
    risk: 'जोखिम',
    disclaimer:
      'यह शोध निर्णय-सहायता परिणाम है। अंतिम निदान योग्य त्वचा विशेषज्ञ द्वारा किया जाना चाहिए।',
    preprocessing: 'प्रीप्रोसेसिंग',
    segmentation: 'सेगमेंटेशन',
    gradcam: 'Grad-CAM',
    enhanced: 'बेहतर किया गया चित्र',
    lesionMask: 'घाव मास्क',
    connect: 'AI मॉड्यूल से कनेक्ट करें',
    realHeatmap: 'इस prediction के लिए Grad-CAM explanation उपलब्ध नहीं है.',

    records: 'मरीज़ रिकॉर्ड',
    analysisHistory: 'विश्लेषण इतिहास',
    searchHistory: 'मरीज़, ID या भविष्यवाणी खोजें',
    allRisk: 'सभी जोखिम',
    prediction: 'भविष्यवाणी',
    date: 'तारीख',
    noRecords: 'अभी कोई मरीज़ रिकॉर्ड नहीं है',

    newAnalysisTitle: 'त्वचा घाव विश्लेषण',
    newAnalysisText:
      'मरीज़ की जानकारी भरें और AI विश्लेषण के लिए घाव चित्र अपलोड करें।',
    aiReady: '● AI सेवा तैयार',
    existing: 'पुराना मरीज़',
    newPatient: 'नया मरीज़',
    existingHelp:
      'सहेजी गई प्रोफ़ाइल पाने के लिए Patient ID दर्ज करें।',
    newHelp:
      'पहली भविष्यवाणी सेव होने पर नया Patient ID बनेगा।',
    patientId: 'Patient ID',
    fetchPatient: 'मरीज़ खोजें',
    fetched: 'मरीज़ प्रोफ़ाइल मिल गई।',
    notFound:
      'Patient ID नहीं मिला। ID जांचें या नया मरीज़ बनाएं।',
    newPatientId: 'नया Patient ID',
    patientProfile: 'मरीज़ और क्लिनिकल जानकारी',
    metadataNote:
      'जानकारी मरीज़ रिकॉर्ड के साथ सेव होगी। वर्तमान image model घाव चित्र से भविष्यवाणी करता है।',
    lesionImage: 'घाव चित्र',
    change: 'बदलें',
    remove: 'हटाएं',
    uploadSkin: 'त्वचा घाव चित्र अपलोड करें',
    drag: 'खींचें और छोड़ें या ब्राउज़ करने के लिए क्लिक करें',
    jpg: 'JPG, JPEG, PNG • अधिकतम 5 MB',
    patientLinked: 'मरीज़ से लिंक',
    readyAnalyze: 'विश्लेषण के लिए तैयार?',
    complete: 'आवश्यक जानकारी और चित्र पूरा करें।',
    allInputs: 'सभी आवश्यक जानकारी उपलब्ध है।',
    analyzing: 'विश्लेषण हो रहा है...',
    analyzeAI: 'AI से विश्लेषण',
    back: 'वापस',

    patientName: 'मरीज़ का नाम *',
    age: 'उम्र *',
    gender: 'लिंग *',
    bodyLocation: 'शरीर का स्थान *',
    duration: 'घाव की अवधि',
    size: 'घाव का आकार',
    colour: 'घाव का रंग',
    shape: 'घाव का आकार/आकृति',
    symptoms: 'लक्षण',
    family: 'पारिवारिक इतिहास',
    sun: 'धूप का संपर्क',
    previous: 'पिछला त्वचा कैंसर',
    select: 'चुनें',
    female: 'महिला',
    male: 'पुरुष',
    other: 'अन्य',
    face: 'चेहरा',
    scalp: 'सिर की त्वचा',
    neck: 'गर्दन',
    chest: 'छाती',
    backBody: 'पीठ',
    arm: 'बांह',
    leg: 'पैर',
    hand: 'हाथ',
    foot: 'पैर का पंजा',
    itching: 'खुजली, खून, दर्द...',

    low: 'कम',
    medium: 'मध्यम',
    high: 'उच्च',
    suspicious: 'संदिग्ध वर्ग',
    nonSuspicious: 'गैर-संदिग्ध वर्ग',
    model: 'मॉडल',
    device: 'डिवाइस',
    probabilities: 'वर्ग संभावनाएं',

    reportCenter: 'रिपोर्ट केंद्र',
    clinicalReport: 'क्लिनिकल-शैली रिपोर्ट',
    print: 'प्रिंट / PDF सेव करें',
    pending: 'लंबित',
    clinicalDisclaimer: 'क्लिनिकल अस्वीकरण',
    reportText:
      'यह सिस्टम शोध और निर्णय-सहायता के लिए है। यह चिकित्सकीय जांच, डर्मोस्कोपी, बायोप्सी या योग्य त्वचा विशेषज्ञ के निदान का विकल्प नहीं है।',

    modelMonitoring: 'मॉडल मॉनिटरिंग',
    analyticsText:
      'डेमो आंकड़ों की जगह प्रशिक्षित मॉडल के वास्तविक मूल्यांकन आंकड़े उपयोग करें।',
    accuracy: 'सटीकता',
    precision: 'प्रिसीजन',
    recall: 'रिकॉल',
    f1: 'F1 स्कोर',
    currentBackend: 'वर्तमान backend',
    apiBase: 'API base URL',
    predictionEndpoint: 'Prediction endpoint',
    live: 'Live AI API',
    privacy: 'गोपनीयता',
    privacyText:
      'केवल अधिकृत मरीज़ डेटा उपयोग करें। उत्पादन में संवेदनशील जानकारी सुरक्षित रखें।',

    languageHint: 'अपने लिए सुविधाजनक भाषा चुनें।',
    invalidImage: 'कृपया एक चित्र फ़ाइल चुनें।',
    max5mb: 'फ़ाइल का अधिकतम आकार 5 MB है।',
    unknown: 'पता नहीं',
    liveSessionRecords: 'लाइव सत्र रिकॉर्ड',
    noRecordsYet: 'अभी कोई रिकॉर्ड नहीं है',
    currentTestAccuracy: 'वर्तमान मॉडल परीक्षण सटीकता',
    weightedPrecision: 'Weighted precision',
    weightedRecall: 'Weighted recall',
    weightedF1: 'Weighted F1',
    lesionAnalysis: 'घाव विश्लेषण',
    explainability: 'व्याख्येयता',
    clinicalReportStep: 'क्लिनिकल रिपोर्ट',
    pipeline1: 'आकार बदलें • सामान्यीकरण • आर्टिफैक्ट हैंडलिंग',
    pipeline2: 'वर्गीकरण + सेगमेंटेशन',
    pipeline3: 'Grad-CAM / ध्यान',
    pipeline4: 'भविष्यवाणी + मरीज़ रिकॉर्ड',
    skinLesionReport: 'त्वचा घाव विश्लेषण रिपोर्ट',
    researchReport: 'शोध निर्णय-सहायता रिपोर्ट'
  },

  mr: {
    dashboard: 'डॅशबोर्ड',
    newPrediction: 'नवीन अंदाज',
    history: 'रुग्ण इतिहास',
    reports: 'रिपोर्ट',
    analytics: 'विश्लेषण',
    settings: 'सेटिंग्स',
    logout: 'लॉगआउट',
    search: 'रुग्ण आणि रिपोर्ट शोधा...',
    researcher: 'संशोधक',
    aiService: 'AI सेवा',
    ready: 'विश्लेषणासाठी तयार',
    language: 'भाषा',
    english: 'English',
    hindi: 'हिंदी',
    marathi: 'मराठी',

    overview: 'आढावा',
    goodAfternoon: 'नमस्कार, Vaishnavi',
    monitor: 'एका ठिकाणाहून AI त्वचा-व्रण विश्लेषण पहा.',
    newAnalysis: 'नवीन विश्लेषण',
    totalAnalyses: 'एकूण विश्लेषणे',
    highRisk: 'उच्च-जोखीम निकाल',
    benign: 'सौम्य / कमी जोखीम',
    validation: 'Validation accuracy',
    recent: 'अलीकडील विश्लेषणे',
    viewAll: 'सर्व पहा',
    riskDistribution: 'जोखीम वितरण',
    aiPipeline: 'AI प्रक्रिया',
    quickActions: 'जलद कृती',
    uploadImage: 'चित्र अपलोड करा',
    patientHistory: 'रुग्ण इतिहास',
    generateReport: 'रिपोर्ट तयार करा',
    modelAnalytics: 'मॉडेल विश्लेषण',

    welcome: 'स्वागत आहे',
    signIn: 'तुमच्या DermaVision workspace मध्ये साइन इन करा',
    email: 'ईमेल',
    password: 'पासवर्ड',
    remember: 'मला लक्षात ठेवा',
    forgot: 'पासवर्ड विसरलात?',
    login: 'लॉगिन',
    demo: 'डेमो मोड: कोणताही valid ईमेल/पासवर्ड डॅशबोर्ड उघडतो.',

    analysisResult: 'विश्लेषण निकाल',
    aiResult: 'AI-सहाय्यित निकाल',
    review: 'भविष्यवाणी, चित्र आणि स्पष्टीकरण पहा.',
    report: 'रिपोर्ट',
    uploaded: 'अपलोड केलेले त्वचा-व्रण चित्र',
    patient: 'रुग्ण',
    aiPrediction: 'AI अंदाज',
    confidence: 'विश्वास पातळी',
    risk: 'जोखीम',
    disclaimer:
      'हा संशोधनासाठी निर्णय-सहाय्य निकाल आहे. अंतिम निदान पात्र त्वचारोगतज्ज्ञाने करावे.',
    preprocessing: 'प्रीप्रोसेसिंग',
    segmentation: 'सेगमेंटेशन',
    gradcam: 'Grad-CAM',
    enhanced: 'Enhanced image',
    lesionMask: 'Lesion mask',
    connect: 'AI मॉड्यूलशी जोडा',
    realHeatmap: 'या prediction साठी Grad-CAM explanation उपलब्ध नाही.',

    records: 'रुग्ण नोंदी',
    analysisHistory: 'विश्लेषण इतिहास',
    searchHistory: 'रुग्ण, Patient ID किंवा अंदाज शोधा',
    allRisk: 'सर्व जोखीम',
    prediction: 'अंदाज',
    date: 'तारीख',
    noRecords: 'अजून रुग्ण नोंदी नाहीत',

    newAnalysisTitle: 'त्वचेच्या व्रणाचे विश्लेषण',
    newAnalysisText:
      'रुग्णाची माहिती भरा आणि AI विश्लेषणासाठी त्वचेचे चित्र अपलोड करा.',
    aiReady: '● AI सेवा तयार',
    existing: 'जुना रुग्ण',
    newPatient: 'नवीन रुग्ण',
    existingHelp:
      'जतन केलेली रुग्ण प्रोफाइल मिळवण्यासाठी Patient ID टाका.',
    newHelp:
      'पहिला अंदाज सेव्ह झाल्यावर नवीन Patient ID तयार होईल.',
    patientId: 'Patient ID',
    fetchPatient: 'रुग्ण शोधा',
    fetched: 'रुग्णाची प्रोफाइल मिळाली.',
    notFound:
      'Patient ID सापडला नाही. ID तपासा किंवा नवीन रुग्ण तयार करा.',
    newPatientId: 'नवीन Patient ID',
    patientProfile: 'रुग्ण आणि क्लिनिकल माहिती',
    metadataNote:
      'माहिती रुग्णाच्या नोंदीसोबत सेव्ह होईल. सध्याचे image model त्वचेच्या चित्रावरून अंदाज करते.',
    lesionImage: 'त्वचेचे चित्र',
    change: 'बदला',
    remove: 'काढा',
    uploadSkin: 'त्वचेचे व्रण चित्र अपलोड करा',
    drag: 'Drag & drop करा किंवा browse करण्यासाठी क्लिक करा',
    jpg: 'JPG, JPEG, PNG • कमाल 5 MB',
    patientLinked: 'रुग्णाशी लिंक',
    readyAnalyze: 'विश्लेषणासाठी तयार?',
    complete: 'आवश्यक माहिती आणि चित्र पूर्ण करा.',
    allInputs: 'सर्व आवश्यक माहिती उपलब्ध आहे.',
    analyzing: 'विश्लेषण सुरू आहे...',
    analyzeAI: 'AI ने विश्लेषण करा',
    back: 'मागे',

    patientName: 'रुग्णाचे नाव *',
    age: 'वय *',
    gender: 'लिंग *',
    bodyLocation: 'शरीराचा भाग *',
    duration: 'व्रणाचा कालावधी',
    size: 'व्रणाचा आकार',
    colour: 'व्रणाचा रंग',
    shape: 'व्रणाची आकृती',
    symptoms: 'लक्षणे',
    family: 'कौटुंबिक इतिहास',
    sun: 'सूर्यप्रकाशाचा संपर्क',
    previous: 'पूर्वीचा त्वचा कर्करोग',
    select: 'निवडा',
    female: 'स्त्री',
    male: 'पुरुष',
    other: 'इतर',
    face: 'चेहरा',
    scalp: 'टाळू',
    neck: 'मान',
    chest: 'छाती',
    backBody: 'पाठ',
    arm: 'हाताचा वरचा भाग',
    leg: 'पाय',
    hand: 'हात',
    foot: 'पाऊल',
    itching: 'खाज, रक्त, वेदना...',

    low: 'कमी',
    medium: 'मध्यम',
    high: 'उच्च',
    suspicious: 'संशयास्पद वर्ग',
    nonSuspicious: 'गैर-संशयास्पद वर्ग',
    model: 'मॉडेल',
    device: 'डिव्हाइस',
    probabilities: 'वर्ग संभाव्यता',

    reportCenter: 'रिपोर्ट केंद्र',
    clinicalReport: 'क्लिनिकल-शैली रिपोर्ट',
    print: 'प्रिंट / PDF सेव्ह करा',
    pending: 'प्रलंबित',
    clinicalDisclaimer: 'क्लिनिकल सूचना',
    reportText:
      'ही प्रणाली संशोधन आणि निर्णय-सहाय्यासाठी आहे. ती वैद्यकीय तपासणी, डर्मोस्कोपी, बायोप्सी किंवा पात्र त्वचारोगतज्ज्ञाच्या निदानाचा पर्याय नाही.',

    modelMonitoring: 'मॉडेल मॉनिटरिंग',
    analyticsText:
      'डेमो आकडे बदलून प्रशिक्षित मॉडेलचे वास्तविक evaluation metrics वापरा.',
    accuracy: 'Accuracy',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1 Score',
    currentBackend: 'सध्याचे backend',
    apiBase: 'API base URL',
    predictionEndpoint: 'Prediction endpoint',
    live: 'Live AI API',
    privacy: 'गोपनीयता',
    privacyText:
      'फक्त अधिकृत रुग्ण डेटा वापरा. उत्पादन प्रणालीमध्ये संवेदनशील माहिती सुरक्षित ठेवा.',

    languageHint: 'तुमच्यासाठी सोयीची भाषा निवडा.',
    invalidImage: 'कृपया चित्र फाइल निवडा.',
    max5mb: 'फाइलचा कमाल आकार 5 MB आहे.',
    unknown: 'माहित नाही',
    liveSessionRecords: 'लाइव्ह सत्र नोंदी',
    noRecordsYet: 'अजून नोंदी नाहीत',
    currentTestAccuracy: 'सध्याच्या मॉडेलची चाचणी अचूकता',
    weightedPrecision: 'Weighted precision',
    weightedRecall: 'Weighted recall',
    weightedF1: 'Weighted F1',
    lesionAnalysis: 'व्रण विश्लेषण',
    explainability: 'स्पष्टीकरणक्षमता',
    clinicalReportStep: 'क्लिनिकल रिपोर्ट',
    pipeline1: 'आकार बदला • सामान्यीकरण • आर्टिफॅक्ट हाताळणी',
    pipeline2: 'वर्गीकरण + सेगमेंटेशन',
    pipeline3: 'Grad-CAM / लक्ष केंद्रित करणे',
    pipeline4: 'अंदाज + रुग्ण नोंद',
    skinLesionReport: 'त्वचेच्या व्रणाचे विश्लेषण रिपोर्ट',
    researchReport: 'संशोधन निर्णय-सहाय्य रिपोर्ट'
  }
};

const conditionNames = {
  'Actinic Keratosis / Bowen Disease': {
    en: 'Actinic Keratosis / Bowen Disease',
    hi: 'एक्टिनिक केराटोसिस / बोवेन रोग',
    mr: 'अॅक्टिनिक केराटोसिस / बोवेन रोग'
  },

  'Basal Cell Carcinoma': {
    en: 'Basal Cell Carcinoma',
    hi: 'बेसल सेल कार्सिनोमा',
    mr: 'बेसल सेल कार्सिनोमा'
  },

  'Benign Keratosis': {
    en: 'Benign Keratosis',
    hi: 'सौम्य केराटोसिस',
    mr: 'सौम्य केराटोसिस'
  },

  Dermatofibroma: {
    en: 'Dermatofibroma',
    hi: 'डर्माटोफायब्रोमा',
    mr: 'डर्माटोफायब्रोमा'
  },

  'Melanocytic Nevus': {
    en: 'Melanocytic Nevus',
    hi: 'मेलानोसाइटिक नेवस',
    mr: 'मेलानोसाइटिक नेवस'
  },

  Melanoma: {
    en: 'Melanoma',
    hi: 'मेलानोमा',
    mr: 'मेलानोमा'
  },

  'Vascular Lesion': {
    en: 'Vascular Lesion',
    hi: 'वास्कुलर घाव',
    mr: 'व्हॅस्क्युलर व्रण'
  }
};

const optionNames = {
  '< 1 month': {
    en: '< 1 month',
    hi: '1 महीने से कम',
    mr: '1 महिन्यापेक्षा कमी'
  },

  '1–6 months': {
    en: '1–6 months',
    hi: '1–6 महीने',
    mr: '1–6 महिने'
  },

  '6–12 months': {
    en: '6–12 months',
    hi: '6–12 महीने',
    mr: '6–12 महिने'
  },

  '> 1 year': {
    en: '> 1 year',
    hi: '1 वर्ष से अधिक',
    mr: '1 वर्षापेक्षा जास्त'
  },

  '< 5 mm': {
    en: '< 5 mm',
    hi: '< 5 मिमी',
    mr: '< 5 मिमी'
  },

  '5–10 mm': {
    en: '5–10 mm',
    hi: '5–10 मिमी',
    mr: '5–10 मिमी'
  },

  '10–20 mm': {
    en: '10–20 mm',
    hi: '10–20 मिमी',
    mr: '10–20 मिमी'
  },

  '> 20 mm': {
    en: '> 20 mm',
    hi: '> 20 मिमी',
    mr: '> 20 मिमी'
  },

  Brown: {
    en: 'Brown',
    hi: 'भूरा',
    mr: 'तपकिरी'
  },

  Black: {
    en: 'Black',
    hi: 'काला',
    mr: 'काळा'
  },

  Red: {
    en: 'Red',
    hi: 'लाल',
    mr: 'लाल'
  },

  Mixed: {
    en: 'Mixed',
    hi: 'मिश्रित',
    mr: 'मिश्रित'
  },

  'Skin-coloured': {
    en: 'Skin-coloured',
    hi: 'त्वचा के रंग का',
    mr: 'त्वचेच्या रंगाचा'
  },

  Regular: {
    en: 'Regular',
    hi: 'नियमित',
    mr: 'नियमित'
  },

  Irregular: {
    en: 'Irregular',
    hi: 'अनियमित',
    mr: 'अनियमित'
  },

  Asymmetric: {
    en: 'Asymmetric',
    hi: 'असममित',
    mr: 'असममित'
  },

  No: {
    en: 'No',
    hi: 'नहीं',
    mr: 'नाही'
  },

  Yes: {
    en: 'Yes',
    hi: 'हाँ',
    mr: 'होय'
  },

  Unknown: {
    en: 'Unknown',
    hi: 'पता नहीं',
    mr: 'माहित नाही'
  },

  Low: {
    en: 'Low',
    hi: 'कम',
    mr: 'कमी'
  },

  Moderate: {
    en: 'Moderate',
    hi: 'मध्यम',
    mr: 'मध्यम'
  },

  High: {
    en: 'High',
    hi: 'उच्च',
    mr: 'उच्च'
  }
};

const optionLabel = (lang, value) =>
  optionNames[value]?.[lang] || value;

const localeFor = {
  en: 'en-IN',
  hi: 'hi-IN',
  mr: 'mr-IN'
};

const tr = (lang, key) =>
  translations[lang]?.[key] ||
  translations.en[key] ||
  key;

const condition = (lang, name) =>
  conditionNames[name]?.[lang] || name;

const riskLabel = (lang, r) =>
  tr(lang, String(r || '').toLowerCase());

function readPatients() {
  try {
    return JSON.parse(
      localStorage.getItem(STORAGE_PATIENTS) || '{}'
    );
  } catch {
    return {};
  }
}

function writePatients(v) {
  localStorage.setItem(
    STORAGE_PATIENTS,
    JSON.stringify(v)
  );
}

function newPatientId() {
  const patients = readPatients();

  let max = 0;

  Object.keys(patients).forEach(id => {
    const m = id.match(/DV-P-(\d+)/);

    if (m) {
      max = Math.max(max, Number(m[1]));
    }
  });

  return `DV-P-${String(max + 1).padStart(5, '0')}`;
}

function formatDate(
  d = new Date(),
  lang = 'en'
) {
  return d.toLocaleDateString(
    localeFor[lang] || 'en-IN',
    {
      day: '2-digit',
      month: 'short',
      year: 'numeric'
    }
  );
}

function LanguageSelect({
  lang,
  setLang
}) {
  return (
    <label className="language-select">
      <span>{tr(lang, 'language')}</span>

      <select
        value={lang}
        onChange={e =>
          setLang(e.target.value)
        }
      >
        <option value="en">
          {tr(lang, 'english')}
        </option>

        <option value="hi">
          {tr(lang, 'hindi')}
        </option>

        <option value="mr">
          {tr(lang, 'marathi')}
        </option>
      </select>
    </label>
  );
}

function Login({
  login,
  lang,
  setLang
}) {
  const [email, setEmail] =
    useState('');

  const [pass, setPass] =
    useState('');

  return (
    <div className="login">

      <div className="login-left">

        <div className="login-brand">

          <span className="logo">
            <I n="shield" />
          </span>

          <b>DermaVision AI</b>

          <LanguageSelect
            lang={lang}
            setLang={setLang}
          />

        </div>

        <div className="hero">

          <small>
            EXPLAINABLE MEDICAL AI
          </small>

          <h1>
            Smarter skin lesion analysis.
            <br />
            <em>
              Clearer AI explanations.
            </em>
          </h1>

          <p>
            Analyze a skin lesion image
            together with patient metadata
            and generate an AI-assisted
            classification, confidence score
            and explainability report.
          </p>

          <div className="chips">
            <span>✓ Image analysis</span>
            <span>✓ Metadata fusion</span>
            <span>✓ Grad-CAM</span>
          </div>

        </div>

      </div>

      <form
        className="login-card"
        onSubmit={e => {
          e.preventDefault();
          login();
        }}
      >

        <div className="login-icon">
          <I n="shield" s={30} />
        </div>

        <h2>
          {tr(lang, 'welcome')}
        </h2>

        <p>
          {tr(lang, 'signIn')}
        </p>

        <label>
          {tr(lang, 'email')}
        </label>

        <input
          type="email"
          value={email}
          onChange={e =>
            setEmail(e.target.value)
          }
          placeholder="you@example.com"
          required
        />

        <label>
          {tr(lang, 'password')}
        </label>

        <input
          type="password"
          value={pass}
          onChange={e =>
            setPass(e.target.value)
          }
          placeholder="Enter password"
          required
        />

        <div className="login-row">
          <span>
            ☐ {tr(lang, 'remember')}
          </span>

          <button
            type="button"
            className="link"
          >
            {tr(lang, 'forgot')}
          </button>
        </div>

        <button className="primary full">
          {tr(lang, 'login')}
        </button>

        <div className="demo">
          {tr(lang, 'demo')}
        </div>

      </form>

    </div>
  );
}

function Side({
  page,
  setPage,
  lang
}) {
  const items = [
    ['Dashboard', 'dash'],
    ['New Prediction', 'plus'],
    ['Patient History', 'hist'],
    ['Reports', 'report'],
    ['Analytics', 'chart'],
    ['Settings', 'set']
  ];

  const keys = {
    Dashboard: 'dashboard',
    'New Prediction': 'newPrediction',
    'Patient History': 'history',
    Reports: 'reports',
    Analytics: 'analytics',
    Settings: 'settings'
  };

  return (
    <aside className="side">

      <div className="brand">

        <span className="logo">
          <I n="shield" />
        </span>

        <div>
          <b>DermaVision AI</b>
          <small>
            SKIN LESION ANALYSIS
          </small>
        </div>

      </div>

      <nav>

        {items.map(item => (
          <button
            className={
              page === item[0]
                ? 'active'
                : ''
            }
            onClick={() =>
              setPage(item[0])
            }
            key={item[0]}
          >

            <I n={item[1]} />

            {tr(
              lang,
              keys[item[0]]
            )}

          </button>
        ))}

      </nav>

      <div className="side-bottom">

        <div className="ready">

          <i />

          <div>
            <b>
              {tr(lang, 'aiService')}
            </b>

            <small>
              {tr(lang, 'ready')}
            </small>
          </div>

        </div>

        <button
          onClick={() =>
            setPage('Login')
          }
        >
          <I n="logout" />

          {tr(lang, 'logout')}
        </button>

      </div>

    </aside>
  );
}

function Top({
  title,
  lang,
  setLang
}) {
  const titleKey = {
    Dashboard: 'dashboard',
    'New Prediction':
      'newPrediction',
    'Patient History':
      'history',
    Reports: 'reports',
    Analytics: 'analytics',
    Settings: 'settings',
    'Prediction Result':
      'analysisResult'
  }[title];

  return (
    <header className="top">

      <b>
        {tr(
          lang,
          titleKey || title
        )}
      </b>

      <div className="top-right">

        <div className="search">

          <I
            n="search"
            s={15}
          />

          <input
            placeholder={tr(
              lang,
              'search'
            )}
          />

        </div>

        <LanguageSelect
          lang={lang}
          setLang={setLang}
        />

        <I
          n="bell"
          s={18}
        />

        <div className="avatar">
          V
        </div>

        <div className="user">
          <b>Vaishnavi</b>
          <small>
            {tr(
              lang,
              'researcher'
            )}
          </small>
        </div>

      </div>

    </header>
  );
}

function Layout({
  page,
  setPage,
  children,
  lang,
  setLang
}) {
  return (
    <div className="shell">

      <Side
        page={page}
        setPage={setPage}
        lang={lang}
      />

      <main>

        <Top
          title={page}
          lang={lang}
          setLang={setLang}
        />

        <div className="content">
          {children}
        </div>

      </main>

    </div>
  );
}

function Metric({
  label,
  value,
  sub
}) {
  return (
    <div className="metric">

      <small>{label}</small>

      <strong>{value}</strong>

      <span>{sub}</span>

    </div>
  );
}

function Dashboard({
  go,
  history,
  lang
}) {
  const total = history.length;

  const high =
    history.filter(
      x => x.risk === 'High'
    ).length;

  const low =
    history.filter(
      x => x.risk === 'Low'
    ).length;

  return (
    <>
      <div className="heading">

        <div>

          <small>
            {tr(lang, 'overview')}
          </small>

          <h1>
            {tr(
              lang,
              'goodAfternoon'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'monitor'
            )}
          </p>

        </div>

        <button
          className="primary"
          onClick={() =>
            go('New Prediction')
          }
        >
          <I n="plus" />

          {tr(
            lang,
            'newAnalysis'
          )}
        </button>

      </div>

      <div className="metrics">

        <Metric
          label={tr(
            lang,
            'totalAnalyses'
          )}
          value={total}
          sub={tr(
            lang,
            'liveSessionRecords'
          )}
        />

        <Metric
          label={tr(
            lang,
            'highRisk'
          )}
          value={high}
          sub={
            total
              ? `${(
                  (high / total) *
                  100
                ).toFixed(1)}% of records`
              : tr(
                  lang,
                  'noRecordsYet'
                )
          }
        />

        <Metric
          label={tr(
            lang,
            'benign'
          )}
          value={low}
          sub={
            total
              ? `${(
                  (low / total) *
                  100
                ).toFixed(1)}% of records`
              : tr(
                  lang,
                  'noRecordsYet'
                )
          }
        />

        <Metric
          label={tr(
            lang,
            'validation'
          )}
          value="61.54%"
          sub={tr(
            lang,
            'currentTestAccuracy'
          )}
        />

      </div>

      <div className="grid">

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'recent'
              )}
            </b>

            <button
              className="link"
              onClick={() =>
                go('Patient History')
              }
            >
              {tr(
                lang,
                'viewAll'
              )}
            </button>

          </div>

          <div className="table">

            {history.length ? (
              <table>

                <thead>

                  <tr>
                    <th>ID</th>
                    <th>
                      {tr(
                        lang,
                        'patient'
                      )}
                    </th>
                    <th>
                      {tr(
                        lang,
                        'prediction'
                      )}
                    </th>
                    <th>
                      {tr(
                        lang,
                        'confidence'
                      )}
                    </th>
                    <th>
                      {tr(
                        lang,
                        'risk'
                      )}
                    </th>
                    <th>
                      {tr(
                        lang,
                        'date'
                      )}
                    </th>
                  </tr>

                </thead>

                <tbody>

                  {history
                    .slice(0, 5)
                    .map(r => (
                      <tr key={r.id}>

                        <td>
                          {r.id}
                        </td>

                        <td>
                          {r.patient}
                        </td>

                        <td>
                          {condition(
                            lang,
                            r.condition
                          )}
                        </td>

                        <td>
                          {Number(
                            r.confidence
                          ).toFixed(2)}
                          %
                        </td>

                        <td>
                          <span
                            className={
                              'risk ' +
                              String(
                                r.risk
                              ).toLowerCase()
                            }
                          >
                            {riskLabel(
                              lang,
                              r.risk
                            )}
                          </span>
                        </td>

                        <td>
                          {r.date}
                        </td>

                      </tr>
                    ))}

                </tbody>

              </table>
            ) : (
              <div className="empty-mini">
                {tr(
                  lang,
                  'noRecords'
                )}
              </div>
            )}

          </div>

        </section>

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'riskDistribution'
              )}
            </b>

          </div>

          <div className="riskbox">

            <div className="donut" />

            <div className="legend">

              <span>
                ● {tr(lang, 'high')}
                <b>
                  {
                    total
                      ? Math.round(
                          (high /
                            total) *
                            100
                        )
                      : 0
                  }%
                </b>
              </span>

              <span>
                ● {tr(lang, 'medium')}
                <b>
                  {
                    total
                      ? Math.round(
                          (history.filter(
                            x =>
                              x.risk ===
                              'Medium'
                          ).length /
                            total) *
                            100
                        )
                      : 0
                  }%
                </b>
              </span>

              <span>
                ● {tr(lang, 'low')}
                <b>
                  {
                    total
                      ? Math.round(
                          (low /
                            total) *
                            100
                        )
                      : 0
                  }%
                </b>
              </span>

            </div>

          </div>

        </section>

      </div>

      <div className="grid bottom">

        <section className="card">

          <div className="card-head">
            <b>
              {tr(
                lang,
                'aiPipeline'
              )}
            </b>
          </div>

          <div className="pipeline">

            {[
              [
                '01',
                tr(
                  lang,
                  'preprocessing'
                ),
                tr(
                  lang,
                  'pipeline1'
                )
              ],
              [
                '02',
                tr(
                  lang,
                  'lesionAnalysis'
                ),
                tr(
                  lang,
                  'pipeline2'
                )
              ],
              [
                '03',
                tr(
                  lang,
                  'explainability'
                ),
                tr(
                  lang,
                  'pipeline3'
                )
              ],
              [
                '04',
                tr(
                  lang,
                  'clinicalReportStep'
                ),
                tr(
                  lang,
                  'pipeline4'
                )
              ]
            ].map(x => (
              <div key={x[0]}>

                <strong>
                  {x[0]}
                </strong>

                <b>
                  {x[1]}
                </b>

                <small>
                  {x[2]}
                </small>

              </div>
            ))}

          </div>

        </section>

        <section className="card">

          <div className="card-head">
            <b>
              {tr(
                lang,
                'quickActions'
              )}
            </b>
          </div>

          <div className="quick">

            <button
              onClick={() =>
                go('New Prediction')
              }
            >
              <I n="upload" />
              {tr(
                lang,
                'uploadImage'
              )}
            </button>

            <button
              onClick={() =>
                go('Patient History')
              }
            >
              <I n="hist" />
              {tr(
                lang,
                'patientHistory'
              )}
            </button>

            <button
              onClick={() =>
                go('Reports')
              }
            >
              <I n="report" />
              {tr(
                lang,
                'generateReport'
              )}
            </button>

            <button
              onClick={() =>
                go('Analytics')
              }
            >
              <I n="chart" />
              {tr(
                lang,
                'modelAnalytics'
              )}
            </button>

          </div>

        </section>

      </div>
    </>
  );
}

function Metadata({
  data,
  setData,
  lang
}) {
  const up = k => e =>
    setData(p => ({
      ...p,
      [k]: e.target.value
    }));

  const fields = [
    [
      'name',
      'patientName',
      'text'
    ],
    [
      'age',
      'age',
      'number'
    ],
    [
      'gender',
      'gender',
      'select:Female|Male|Other'
    ],
    [
      'location',
      'bodyLocation',
      'select:Face|Scalp|Neck|Chest|Back|Arm|Leg|Hand|Foot'
    ],
    [
      'duration',
      'duration',
      'select:< 1 month|1–6 months|6–12 months|> 1 year'
    ],
    [
      'size',
      'size',
      'select:< 5 mm|5–10 mm|10–20 mm|> 20 mm'
    ],
    [
      'color',
      'colour',
      'select:Brown|Black|Red|Mixed|Skin-coloured'
    ],
    [
      'shape',
      'shape',
      'select:Regular|Irregular|Asymmetric'
    ],
    [
      'symptoms',
      'symptoms',
      'text'
    ],
    [
      'family',
      'family',
      'select:No|Yes|Unknown'
    ],
    [
      'sun',
      'sun',
      'select:Low|Moderate|High'
    ],
    [
      'previous',
      'previous',
      'select:No|Yes|Unknown'
    ]
  ];

  return (
    <div className="metadata">

      {fields.map(
        ([k, l, t]) => (
          <div
            className="field"
            key={k}
          >

            <label>
              {tr(lang, l)}
            </label>

            {t.startsWith(
              'select:'
            ) ? (
              <select
                value={data[k]}
                onChange={up(k)}
              >

                <option value="">
                  {tr(
                    lang,
                    'select'
                  )}
                </option>

                {t
                  .slice(7)
                  .split('|')
                  .map(o => (
                    <option
                      key={o}
                      value={o}
                    >
                      {optionLabel(
                        lang,
                        o
                      )}
                    </option>
                  ))}

              </select>
            ) : (
              <input
                type={t}
                value={data[k]}
                onChange={up(k)}
                placeholder={
                  k === 'symptoms'
                    ? tr(
                        lang,
                        'itching'
                      )
                    : ''
                }
              />
            )}

          </div>
        )
      )}

    </div>
  );
}

function Uploader({
  preview,
  setPreview,
  file,
  setFile,
  lang
}) {
  const ref = useRef();

  const pick = f => {

    if (!f) return;

    if (
      !f.type.startsWith(
        'image/'
      )
    ) {
      return alert(
        tr(
          lang,
          'invalidImage'
        )
      );
    }

    if (
      f.size >
      5 * 1024 * 1024
    ) {
      return alert(
        tr(
          lang,
          'max5mb'
        )
      );
    }

    setFile(f);

    setPreview(
      URL.createObjectURL(f)
    );
  };

  return !preview ? (
    <div
      className="drop"
      onClick={() =>
        ref.current.click()
      }
    >

      <div className="upload-icon">
        <I
          n="upload"
          s={30}
        />
      </div>

      <b>
        {tr(
          lang,
          'uploadSkin'
        )}
      </b>

      <span>
        {tr(
          lang,
          'drag'
        )}
      </span>

      <small>
        {tr(
          lang,
          'jpg'
        )}
      </small>

      <input
        ref={ref}
        hidden
        type="file"
        accept="image/*"
        onChange={e =>
          pick(
            e.target.files?.[0]
          )
        }
      />

    </div>
  ) : (
    <div className="preview">

      <img
        src={preview}
        alt="Uploaded skin lesion"
      />

      <div>

        <b>
          {file?.name}
        </b>

        <small>
          {(
            file.size /
            1024 /
            1024
          ).toFixed(2)}{' '}
          MB
        </small>

        <div>

          <button
            type="button"
            className="secondary"
            onClick={() =>
              ref.current.click()
            }
          >
            {tr(
              lang,
              'change'
            )}
          </button>

          <button
            type="button"
            className="danger"
            onClick={() => {
              setFile(null);
              setPreview(null);
            }}
          >
            {tr(
              lang,
              'remove'
            )}
          </button>

        </div>

      </div>

      <input
        ref={ref}
        hidden
        type="file"
        accept="image/*"
        onChange={e =>
          pick(
            e.target.files?.[0]
          )
        }
      />

    </div>
  );
}

function NewPrediction({
  setResult,
  addHistory,
  go,
  lang
}) {
  const [mode, setMode] =
    useState('new');

  const [data, setData] =
    useState(initial);

  const [patientId, setPatientId] =
    useState('');

  const [lookupId, setLookupId] =
    useState('');

  const [file, setFile] =
    useState(null);

  const [preview, setPreview] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [message, setMessage] =
    useState('');

  const [error, setError] =
    useState('');

  const valid = useMemo(
    () =>
      data.name &&
      data.age &&
      data.gender &&
      data.location &&
      file,
    [data, file]
  );

  const fetchPatient = () => {

    setError('');
    setMessage('');

    const patients =
      readPatients();

    const p =
      patients[
        lookupId.trim()
      ];

    if (!p) {

      setError(
        tr(
          lang,
          'notFound'
        )
      );

      return;
    }

    setPatientId(p.id);

    setData({
      ...initial,
      ...p.profile
    });

    setMessage(
      tr(
        lang,
        'fetched'
      )
    );
  };

  const analyze = async () => {

    if (!valid) {
      return alert(
        tr(
          lang,
          'complete'
        )
      );
    }

    setLoading(true);
    setError('');

    try {

      // ============================
      // 1. MAIN AI PREDICTION
      // ============================

      const predictForm =
        new FormData();

      predictForm.append(
        'image',
        file
      );

      const res =
        await fetch(
          `${API_BASE}/predict`,
          {
            method: 'POST',
            body: predictForm
          }
        );

      const body =
        await res.json()
          .catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          body.error ||
          `Backend error ${res.status}`
        );
      }

      // ============================
      // 2. REAL IMAGE PREPROCESSING
      // ============================

      const preprocessForm =
        new FormData();

      preprocessForm.append(
        'image',
        file
      );

      const preprocessRes =
        await fetch(
          `${API_BASE}/preprocess`,
          {
            method: 'POST',
            body: preprocessForm
          }
        );

      const preprocessBody =
        await preprocessRes
          .json()
          .catch(() => ({}));

      if (!preprocessRes.ok) {
        throw new Error(
          preprocessBody.error ||
          `Preprocessing error ${preprocessRes.status}`
        );
      }

      // ============================
      // 3. REAL LESION SEGMENTATION
      // ============================

      let segmentationBody = null;

      try {
        const segmentForm =
          new FormData();

        segmentForm.append(
          'image',
          file
        );

        const segmentRes =
          await fetch(
            `${API_BASE}/segment`,
            {
              method: 'POST',
              body: segmentForm
            }
          );

        segmentationBody =
          await segmentRes
            .json()
            .catch(() => ({}));

        if (!segmentRes.ok) {
          throw new Error(
            segmentationBody.error ||
            `Segmentation error ${segmentRes.status}`
          );
        }
      } catch (segmentationError) {
        console.warn(
          'Segmentation service unavailable:',
          segmentationError
        );

        segmentationBody = {
          available: false,
          error:
            segmentationError?.message ||
            'Segmentation service unavailable.'
        };
      }

      // ============================
      // 4. PATIENT ID
      // ============================

      const id =
        mode === 'existing' &&
        patientId
          ? patientId
          : newPatientId();

      const patients =
        readPatients();

      const old =
        patients[id];

      const profile = {
        ...data
      };

      const record = {
        id,
        profile,
        predictions:
          old?.predictions || []
      };

      record.predictions.unshift({
        condition:
          body.condition ||
          body.prediction,

        confidence:
          Number(
            body.confidence || 0
          ),

        risk:
          body.risk ||
          'Medium',

        predictedClass:
          body.predicted_class,

        createdAt:
          new Date().toISOString()
      });

      patients[id] =
        record;

      writePatients(
        patients
      );

      // ============================
      // 4. FINAL RESULT
      // ============================

      const result = {
        ...body,

        patient: data,

        patientId: id,

        // Original image
        preview,

        // REAL processed image
        preprocessedImage:
          preprocessBody.image,

        // Processing steps
        preprocessingSteps:
          preprocessBody.processing ||
          [],

        // REAL lesion segmentation
        segmentation:
          segmentationBody
      };

      setResult(result);

      addHistory({
        id,
        patient:
          data.name,

        patientId:
          id,

        condition:
          body.condition ||
          body.prediction,

        confidence:
          Number(
            body.confidence || 0
          ),

        risk:
          body.risk ||
          'Medium',

        date:
          formatDate(
            new Date(),
            lang
          )
      });

      setPatientId(id);

      go(
        'Prediction Result'
      );

    } catch (e) {

      console.error(e);

      setError(
        e.message ||
        'Prediction failed.'
      );

    } finally {

      setLoading(false);

    }
  };

  return (
    <>
      <div className="heading">

        <div>

          <small>
            {tr(
              lang,
              'newAnalysis'
            ).toUpperCase()}
          </small>

          <h1>
            {tr(
              lang,
              'newAnalysisTitle'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'newAnalysisText'
            )}
          </p>

        </div>

        <span className="ready-chip">
          {tr(
            lang,
            'aiReady'
          )}
        </span>

      </div>

      <section className="card patient-mode">

        <div className="mode-buttons">

          <button
            className={
              mode === 'new'
                ? 'active'
                : ''
            }
            onClick={() => {

              setMode('new');
              setPatientId('');
              setData(initial);
              setMessage('');
              setError('');

            }}
          >
            {tr(
              lang,
              'newPatient'
            )}
          </button>

          <button
            className={
              mode === 'existing'
                ? 'active'
                : ''
            }
            onClick={() => {

              setMode(
                'existing'
              );

              setPatientId('');
              setMessage('');
              setError('');

            }}
          >
            {tr(
              lang,
              'existing'
            )}
          </button>

        </div>

        {mode === 'existing' ? (

          <div className="lookup">

            <div>

              <label>
                {tr(
                  lang,
                  'patientId'
                )}
              </label>

              <input
                value={lookupId}
                onChange={e =>
                  setLookupId(
                    e.target.value.toUpperCase()
                  )
                }
                placeholder="DV-P-00001"
              />

            </div>

            <button
              className="secondary"
              onClick={
                fetchPatient
              }
            >
              {tr(
                lang,
                'fetchPatient'
              )}
            </button>

            {message && (
              <span className="success-msg">
                ✓ {message}
              </span>
            )}

            {error && (
              <span className="error-msg">
                {error}
              </span>
            )}

          </div>

        ) : (

          <div className="new-id-note">

            <I
              n="info"
              s={17}
            />

            <span>
              {tr(
                lang,
                'newHelp'
              )}
            </span>

          </div>

        )}

        {patientId && (
          <div className="patient-id-chip">

            <b>
              {tr(
                lang,
                'patientId'
              )}
              :
            </b>

            {patientId}

          </div>
        )}

      </section>

      <div className="form-grid">

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'patientProfile'
              )}
            </b>

          </div>

          <p className="note">
            {tr(
              lang,
              'metadataNote'
            )}
          </p>

          <Metadata
            data={data}
            setData={setData}
            lang={lang}
          />

        </section>

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'lesionImage'
              )}
            </b>

          </div>

          <Uploader
            preview={preview}
            setPreview={setPreview}
            file={file}
            setFile={setFile}
            lang={lang}
          />

          <div className="checks">

            <span>
              ✓ JPG / PNG
            </span>

            <span>
              ✓ Max 5 MB
            </span>

            <span>
              ✓ {tr(
                lang,
                'patientLinked'
              )}
            </span>

          </div>

        </section>

      </div>

      {error &&
        mode === 'new' && (
          <div className="api-error">
            {error}
          </div>
        )}

      <section className="card analyze">

        <div>

          <b>
            {tr(
              lang,
              'readyAnalyze'
            )}
          </b>

          <small>
            {valid
              ? tr(
                  lang,
                  'allInputs'
                )
              : tr(
                  lang,
                  'complete'
                )}
          </small>

        </div>

        <button
          disabled={
            !valid ||
            loading
          }
          className="primary"
          onClick={
            analyze
          }
        >

          {loading ? (
            tr(
              lang,
              'analyzing'
            )
          ) : (
            <>
              <I n="shield" />

              {tr(
                lang,
                'analyzeAI'
              )}

              <I
                n="arrow"
                s={16}
              />
            </>
          )}

        </button>

      </section>
    </>
  );
}

function Result({
  result,
  go,
  lang
}) {
  if (!result) {
    return (
      <section className="card empty">

        <I
          n="info"
          s={40}
        />

        <h2>
          {tr(
            lang,
            'pending'
          )}
        </h2>

        <p>
          {tr(
            lang,
            'complete'
          )}
        </p>

        <button
          className="primary"
          onClick={() =>
            go(
              'New Prediction'
            )
          }
        >
          {tr(
            lang,
            'newAnalysis'
          )}
        </button>

      </section>
    );
  }

  const probs =
    result.probabilities ||
    {};

  return (
    <>
      <div className="heading">

        <div>

          <small>
            {tr(
              lang,
              'analysisResult'
            )}
          </small>

          <h1>
            {tr(
              lang,
              'aiResult'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'review'
            )}
          </p>

        </div>

        <button
          className="secondary"
          onClick={() =>
            go('Reports')
          }
        >
          <I n="report" />

          {tr(
            lang,
            'report'
          )}
        </button>

      </div>

      <div className="result-grid">

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'uploaded'
              )}
            </b>

          </div>

          <img
            className="result-image"
            src={result.preview}
            alt="Uploaded lesion"
          />

          <small className="muted">

            {tr(
              lang,
              'patient'
            )}
            : {result.patient.name}

            {' • '}

            {tr(
              lang,
              'patientId'
            )}
            : {result.patientId}

            {' • '}

            {result.patient.location}

          </small>

        </section>

        <section className="prediction">

          <div className="result-top">

            <span>
              {tr(
                lang,
                'aiPrediction'
              )}
            </span>

            <span
              className={
                'risk ' +
                String(
                  result.risk ||
                    'Medium'
                ).toLowerCase()
              }
            >
              {riskLabel(
                lang,
                result.risk ||
                  'Medium'
              )}
            </span>

          </div>

          <h2>
            {condition(
              lang,
              result.condition
            )}
          </h2>

          <div className="confidence">

            <span>
              {tr(
                lang,
                'confidence'
              )}
            </span>

            <b>
              {Number(
                result.confidence
              ).toFixed(2)}
              %
            </b>

          </div>

          <div className="bar">

            <i
              style={{
                width:
                  Math.min(
                    100,
                    Math.max(
                      0,
                      Number(
                        result.confidence
                      )
                    )
                  ) + '%'
              }}
            />

          </div>

          <p>
            {tr(
              lang,
              'disclaimer'
            )}
          </p>

          <div className="result-meta">

            <span>
              <b>
                {tr(
                  lang,
                  'patientId'
                )}
              </b>

              {result.patientId}
            </span>

            <span>
              <b>
                {tr(
                  lang,
                  'model'
                )}
              </b>

              {result.model ||
                'DermaVision EfficientNet-B0'}
            </span>

            <span>
              <b>
                {tr(
                  lang,
                  'device'
                )}
              </b>

              {result.device ||
                'cpu'}
            </span>

          </div>

        </section>

      </div>

      {/* =====================================
          AI EVIDENCE
      ====================================== */}

      <div className="evidence">

        {/* PREPROCESSING */}

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'preprocessing'
              )}
            </b>

          </div>

          {result.preprocessedImage ? (
            <>
              <img
                className="evidence-image"
                src={
                  result.preprocessedImage
                }
                alt="Preprocessed skin lesion"
              />

              <small className="muted">

                {tr(
                  lang,
                  'enhanced'
                )}

                {result.preprocessingSteps?.length
                  ? ' • ' +
                    result.preprocessingSteps.join(
                      ' • '
                    )
                  : ''}

              </small>

            </>
          ) : (

            <div className="placeholder">

              {tr(
                lang,
                'enhanced'
              )}

              <br />

              <small>
                {tr(
                  lang,
                  'connect'
                )}
              </small>

            </div>

          )}

        </section>

        {/* SEGMENTATION */}

        <section className="card segmentation-card">

          <div className="card-head">

            <div>
              <b>
                {tr(
                  lang,
                  'segmentation'
                )}
              </b>

              <small className="segmentation-subtitle">
                {result?.segmentation?.method ||
                  'Lesion region extraction'}
              </small>
            </div>

            {result?.segmentation?.available !== false && (
              <span className="ai-status">
                AI-ready
              </span>
            )}

          </div>

          {result?.segmentation?.overlay ? (

            <div className="segmentation-container">

              <div className="segmentation-image-wrapper">
                <img
                  className="segmentation-image"
                  src={
                    result.segmentation.overlay
                  }
                  alt="Segmented skin lesion"
                />
              </div>

              <div className="segmentation-details">

                <div className="segmentation-condition">

                  <strong>
                    {tr(
                      lang,
                      'lesionMask'
                    )}
                  </strong>

                  {typeof result.segmentation.area_percentage ===
                    'number' && (
                    <span className="segmentation-area">
                      {result.segmentation.area_percentage.toFixed(2)}
                      % area
                    </span>
                  )}

                </div>

                <p>
                  {result.segmentation.message ||
                    'Detected lesion boundary is highlighted on the original image.'}
                </p>

                {result.segmentation.lesion && (
                  <img
                    className="segmentation-lesion-image"
                    src={
                      result.segmentation.lesion
                    }
                    alt="Extracted lesion region"
                  />
                )}

              </div>

            </div>

          ) : (

            <div className="placeholder segmentation-placeholder">

              {tr(
                lang,
                'lesionMask'
              )}

              <br />

              <small>
                {result?.segmentation?.error ||
                  tr(
                    lang,
                    'connect'
                  )}
              </small>

            </div>

          )}

        </section>

        {/* GRAD CAM */}

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'gradcam'
              )}
            </b>

          </div>

          {result?.gradcam?.image ? (
            <div className="gradcam-container">
              <img
                className="gradcam-image"
                src={result.gradcam.image}
                alt="Grad-CAM explanation"
              />

              <div className="gradcam-info">
                <strong>
                  {result.gradcam.condition
                    ? condition(lang, result.gradcam.condition)
                    : tr(lang, 'gradcam')}
                </strong>

                <small className="muted">
                  {result.gradcam.message ||
                    'Real Grad-CAM explanation • highlights regions influencing the AI prediction.'}
                </small>
              </div>
            </div>
          ) : (
            <div className="placeholder">
              {tr(lang, 'gradcam')}
              <br />
              <small>{tr(lang, 'realHeatmap')}</small>
            </div>
          )}

        </section>

      </div>

      {/* PROBABILITIES */}

      <section className="card probability-card">

        <div className="card-head">

          <b>
            {tr(
              lang,
              'probabilities'
            )}
          </b>

        </div>

        {Object.entries(
          probs
        )
          .sort(
            (a, b) =>
              Number(b[1]) -
              Number(a[1])
          )
          .map(
            ([name, val]) => (
              <div
                className="prob-row"
                key={name}
              >

                <span>
                  {condition(
                    lang,
                    name
                  )}
                </span>

                <b>
                  {Number(
                    val
                  ).toFixed(2)}
                  %
                </b>

                <i>

                  <em
                    style={{
                      width:
                        Math.min(
                          100,
                          Number(
                            val
                          )
                        ) + '%'
                    }}
                  />

                </i>

              </div>
            )
          )}

      </section>
    </>
  );
}

function History({
  history,
  go,
  lang
}) {
  const [q, setQ] =
    useState('');

  const [risk, setRisk] =
    useState('All');

  const filtered =
    history.filter(
      r =>
        (
          !q ||
          `${r.id} ${r.patient} ${r.condition}`
            .toLowerCase()
            .includes(
              q.toLowerCase()
            )
        ) &&
        (
          risk === 'All' ||
          r.risk === risk
        )
    );

  return (
    <>
      <div className="heading">

        <div>

          <small>
            {tr(
              lang,
              'records'
            )}
          </small>

          <h1>
            {tr(
              lang,
              'analysisHistory'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'review'
            )}
          </p>

        </div>

        <button
          className="primary"
          onClick={() =>
            go(
              'New Prediction'
            )
          }
        >
          <I n="plus" />

          {tr(
            lang,
            'newAnalysis'
          )}

        </button>

      </div>

      <section className="card">

        <div className="toolbar">

          <div className="search wide">

            <I
              n="search"
              s={15}
            />

            <input
              value={q}
              onChange={e =>
                setQ(
                  e.target.value
                )
              }
              placeholder={tr(
                lang,
                'searchHistory'
              )}
            />

          </div>

          <select
            value={risk}
            onChange={e =>
              setRisk(
                e.target.value
              )
            }
          >

            <option value="All">
              {tr(
                lang,
                'allRisk'
              )}
            </option>

            <option value="High">
              {tr(
                lang,
                'high'
              )}
            </option>

            <option value="Medium">
              {tr(
                lang,
                'medium'
              )}
            </option>

            <option value="Low">
              {tr(
                lang,
                'low'
              )}
            </option>

          </select>

        </div>

        <div className="table">

          {filtered.length ? (

            <table>

              <thead>

                <tr>

                  <th>
                    {tr(
                      lang,
                      'patientId'
                    )}
                  </th>

                  <th>
                    {tr(
                      lang,
                      'patient'
                    )}
                  </th>

                  <th>
                    {tr(
                      lang,
                      'prediction'
                    )}
                  </th>

                  <th>
                    {tr(
                      lang,
                      'confidence'
                    )}
                  </th>

                  <th>
                    {tr(
                      lang,
                      'risk'
                    )}
                  </th>

                  <th>
                    {tr(
                      lang,
                      'date'
                    )}
                  </th>

                </tr>

              </thead>

              <tbody>

                {filtered.map(
                  r => (
                    <tr key={r.id}>

                      <td>
                        <b>
                          {r.patientId ||
                            r.id}
                        </b>
                      </td>

                      <td>
                        {r.patient}
                      </td>

                      <td>
                        {condition(
                          lang,
                          r.condition
                        )}
                      </td>

                      <td>
                        {Number(
                          r.confidence
                        ).toFixed(2)}
                        %
                      </td>

                      <td>

                        <span
                          className={
                            'risk ' +
                            String(
                              r.risk
                            ).toLowerCase()
                          }
                        >
                          {riskLabel(
                            lang,
                            r.risk
                          )}
                        </span>

                      </td>

                      <td>
                        {r.date}
                      </td>

                    </tr>
                  )
                )}

              </tbody>

            </table>

          ) : (

            <div className="empty-mini">
              {tr(
                lang,
                'noRecords'
              )}
            </div>

          )}

        </div>

      </section>
    </>
  );
}

function Analytics({
  lang
}) {
  return (
    <>
      <div className="heading">

        <div>

          <small>
            {tr(
              lang,
              'modelMonitoring'
            )}
          </small>

          <h1>
            {tr(
              lang,
              'analytics'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'analyticsText'
            )}
          </p>

        </div>

      </div>

      <div className="metrics">

        <Metric
          label={tr(
            lang,
            'accuracy'
          )}
          value="61.54%"
          sub="Current test accuracy"
        />

        <Metric
          label={tr(
            lang,
            'precision'
          )}
          value="78.71%"
          sub={tr(
            lang,
            'weightedPrecision'
          )}
        />

        <Metric
          label={tr(
            lang,
            'recall'
          )}
          value="61.54%"
          sub={tr(
            lang,
            'weightedRecall'
          )}
        />

        <Metric
          label={tr(
            lang,
            'f1'
          )}
          value="65.38%"
          sub={tr(
            lang,
            'weightedF1'
          )}
        />

      </div>

      <div className="grid">

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'currentBackend'
              )}
            </b>

          </div>

          <div className="settings">

            <div>

              <span>
                {tr(
                  lang,
                  'apiBase'
                )}
              </span>

              <code>
                {API_BASE}
              </code>

            </div>

            <div>

              <span>
                {tr(
                  lang,
                  'predictionEndpoint'
                )}
              </span>

              <code>
                POST /predict
              </code>

            </div>

            <div>

              <span>
                {tr(
                  lang,
                  'model'
                )}
              </span>

              <em>
                EfficientNet-B0
              </em>

            </div>

            <div>

              <span>
                {tr(
                  lang,
                  'device'
                )}
              </span>

              <em>
                CPU
              </em>

            </div>

          </div>

        </section>

        <section className="card">

          <div className="card-head">

            <b>
              {tr(
                lang,
                'privacy'
              )}
            </b>

          </div>

          <p className="privacy-text">
            {tr(
              lang,
              'privacyText'
            )}
          </p>

        </section>

      </div>
    </>
  );
}

function Reports({
  result,
  lang
}) {
  return (
    <>
      <div className="heading print-hide">

        <div>

          <small>
            {tr(
              lang,
              'reportCenter'
            )}
          </small>

          <h1>
            {tr(
              lang,
              'clinicalReport'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'review'
            )}
          </p>

        </div>

        <button
          className="primary"
          onClick={() =>
            window.print()
          }
        >

          <I n="report" />

          {tr(
            lang,
            'print'
          )}

        </button>

      </div>

      <div className="report">

        <div className="report-head">

          <div>

            <span className="logo">

              <I n="shield" />

            </span>

            <div>

              <b>
                DermaVision AI
              </b>

              <small>
                {tr(
                  lang,
                  'skinLesionReport'
                )}
              </small>

            </div>

          </div>

          <small>
            {tr(
              lang,
              'researchReport'
            )}
          </small>

        </div>

        <h2>
          {tr(
            lang,
            'aiResult'
          )}
        </h2>

        <div className="report-info">

          <div>

            <b>
              {tr(
                lang,
                'patient'
              )}
            </b>

            <span>
              {result?.patient?.name ||
                '—'}
            </span>

          </div>

          <div>

            <b>
              {tr(
                lang,
                'patientId'
              )}
            </b>

            <span>
              {result?.patientId ||
                '—'}
            </span>

          </div>

          <div>

            <b>
              {tr(
                lang,
                'age'
              )}{' '}
              /{' '}
              {tr(
                lang,
                'gender'
              )}
            </b>

            <span>
              {result?.patient?.age ||
                '—'}{' '}
              /{' '}
              {result?.patient?.gender ||
                '—'}
            </span>

          </div>

          <div>

            <b>
              {tr(
                lang,
                'bodyLocation'
              )}
            </b>

            <span>
              {result?.patient?.location ||
                '—'}
            </span>

          </div>

        </div>

        <div className="report-images">

          <div>

            <small>
              {tr(
                lang,
                'uploaded'
              )}
            </small>

            {result?.preview ? (
              <img
                src={
                  result.preview
                }
                alt="Uploaded"
              />
            ) : (
              <div className="report-empty">
                {tr(
                  lang,
                  'pending'
                )}
              </div>
            )}

          </div>

          <div>

            <small>
              {tr(
                lang,
                'preprocessing'
              )}
            </small>

            {result?.preprocessedImage ? (
              <img
                src={
                  result.preprocessedImage
                }
                alt="Preprocessed"
              />
            ) : (
              <div className="report-empty">
                {tr(
                  lang,
                  'pending'
                )}
              </div>
            )}

          </div>

        </div>

        <div className="report-result">

          <div>

            <small>
              {tr(
                lang,
                'prediction'
              )}
            </small>

            <b>
              {result
                ? condition(
                    lang,
                    result.condition
                  )
                : tr(
                    lang,
                    'pending'
                  )}
            </b>

          </div>

          <div>

            <small>
              {tr(
                lang,
                'confidence'
              )}
            </small>

            <b>
              {result
                ? Number(
                    result.confidence
                  ).toFixed(2) +
                  '%'
                : '—'}
            </b>

          </div>

          <div>

            <small>
              {tr(
                lang,
                'risk'
              )}
            </small>

            <b>
              {result
                ? riskLabel(
                    lang,
                    result.risk
                  )
                : '—'}
            </b>

          </div>

        </div>

        <div className="disclaimer">

          <b>
            {tr(
              lang,
              'clinicalDisclaimer'
            )}
          </b>

          <p>
            {tr(
              lang,
              'reportText'
            )}
          </p>

        </div>

      </div>
    </>
  );
}

function Settings({
  lang
}) {
  return (
    <>
      <div className="heading">

        <div>

          <small>
            {tr(
              lang,
              'settings'
            ).toUpperCase()}
          </small>

          <h1>
            {tr(
              lang,
              'settings'
            )}
          </h1>

          <p>
            {tr(
              lang,
              'languageHint'
            )}
          </p>

        </div>

      </div>

      <div className="grid">

        <section className="card settings">

          <b>
            {tr(
              lang,
              'currentBackend'
            )}
          </b>

          <div>

            <span>
              {tr(
                lang,
                'apiBase'
              )}
            </span>

            <code>
              {API_BASE}
            </code>

          </div>

          <div>

            <span>
              {tr(
                lang,
                'predictionEndpoint'
              )}
            </span>

            <code>
              POST /predict
            </code>

          </div>

          <div>

            <span>
              {tr(
                lang,
                'model'
              )}
            </span>

            <em>
              DermaVision EfficientNet-B0
            </em>

          </div>

          <div>

            <span>
              {tr(
                lang,
                'live'
              )}
            </span>

            <em>
              ✓
            </em>

          </div>

        </section>

        <section className="card settings">

          <b>
            {tr(
              lang,
              'privacy'
            )}
          </b>

          <p>
            {tr(
              lang,
              'privacyText'
            )}
          </p>

        </section>

      </div>
    </>
  );
}

function loadHistory() {
  const patients =
    readPatients();

  return Object.values(
    patients
  )
    .flatMap(
      p =>
        (
          p.predictions ||
          []
        ).map(
          (x, i) => ({
            id:
              `${p.id}-${String(
                i + 1
              ).padStart(2, '0')}`,

            patient:
              p.profile?.name ||
              '—',

            patientId:
              p.id,

            condition:
              x.condition,

            confidence:
              Number(
                x.confidence ||
                  0
              ),

            risk:
              x.risk ||
              'Medium',

            date:
              formatDate(
                new Date(
                  x.createdAt ||
                    Date.now()
                ),
                'en'
              )
          })
        )
    )
    .sort(
      (a, b) =>
        b.date.localeCompare(
          a.date
        )
    );
}

export default function App() {

  const [page, setPage] =
    useState('Login');

  const [lang, setLangState] =
    useState(
      () =>
        localStorage.getItem(
          STORAGE_LANGUAGE
        ) || 'en'
    );

  const [history, setHistory] =
    useState(
      loadHistory
    );

  const [result, setResult] =
    useState(null);

  const setLang = value => {

    setLangState(value);

    localStorage.setItem(
      STORAGE_LANGUAGE,
      value
    );

  };

  const go = p =>
    setPage(p);

  const add = r =>
    setHistory(
      h => [r, ...h]
    );

  if (page === 'Login') {

    return (
      <Login
        login={() =>
          go('Dashboard')
        }
        lang={lang}
        setLang={setLang}
      />
    );
  }

  let content;

  if (page === 'Dashboard') {

    content = (
      <Dashboard
        go={go}
        history={history}
        lang={lang}
      />
    );

  } else if (
    page === 'New Prediction'
  ) {

    content = (
      <NewPrediction
        setResult={setResult}
        addHistory={add}
        go={go}
        lang={lang}
      />
    );

  } else if (
    page === 'Prediction Result'
  ) {

    content = (
      <Result
        result={result}
        go={go}
        lang={lang}
      />
    );

  } else if (
    page === 'Patient History'
  ) {

    content = (
      <History
        history={history}
        go={go}
        lang={lang}
      />
    );

  } else if (
    page === 'Analytics'
  ) {

    content = (
      <Analytics
        lang={lang}
      />
    );

  } else if (
    page === 'Reports'
  ) {

    content = (
      <Reports
        result={result}
        lang={lang}
      />
    );

  } else {

    content = (
      <Settings
        lang={lang}
      />
    );
  }

  return (
    <Layout
      page={page}
      setPage={go}
      lang={lang}
      setLang={setLang}
    >
      {content}
    </Layout>
  );
}