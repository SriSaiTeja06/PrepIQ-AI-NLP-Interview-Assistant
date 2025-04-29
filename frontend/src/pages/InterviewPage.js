import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import ApiClient from '../api/client';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Paper from '@mui/material/Paper';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import CircularProgress from '@mui/material/CircularProgress';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';

const steps = ['Select Role', 'Get Question', 'Record Answer', 'View Feedback'];

const InterviewPage = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [roles, setRoles] = useState([]);
  const [selectedRole, setSelectedRole] = useState('');
  const [questionType, setQuestionType] = useState('technical');
  const [difficulty, setDifficulty] = useState('medium');
  const [question, setQuestion] = useState(null);
  const [answer, setAnswer] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [feedback, setFeedback] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  
  useEffect(() => {
    const fetchRoles = async () => {
      try {
        setLoading(true);
        setRoles([
          'Software Engineer',
          'Data Scientist',
          'Machine Learning Engineer',
          'Frontend Developer',
          'Backend Developer',
          'Full Stack Developer',
          'DevOps Engineer',
          'Cloud Engineer',
          'Data Engineer',
          'Security Engineer',
          'QA Engineer',
          'Mobile Developer',
          'Site Reliability Engineer'
        ]);
        setLoading(false);
      } catch (err) {
        setError('Failed to load roles. Please try again.');
        setLoading(false);
      }
    };
    
    fetchRoles();
  }, []);
  
  const handleRoleChange = (event) => {
    setSelectedRole(event.target.value);
  };
  
  const handleTypeChange = (event) => {
    setQuestionType(event.target.value);
  };
  
  const handleDifficultyChange = (event) => {
    setDifficulty(event.target.value);
  };
  
  const handleGetQuestion = async () => {
    try {
      setLoading(true);
      setError(null);
      const questionData = await ApiClient.generateQuestion(selectedRole, difficulty, questionType);
      setQuestion(questionData);
      setLoading(false);
      setActiveStep(1);
    } catch (err) {
      setError('Failed to get question. Please try again.');
      setLoading(false);
    }
  };
  
  const handleStartRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioBlob(audioBlob);
        setAudioUrl(audioUrl);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      setError('Failed to access microphone. Please check your browser permissions.');
    }
  };
  
  const handleStopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };
  
  const handleSpeakQuestion = () => {
    if ('speechSynthesis' in window && question) {
      const utterance = new SpeechSynthesisUtterance(question.content);
      window.speechSynthesis.speak(utterance);
    } else {
      setError('Text-to-speech is not supported in your browser.');
    }
  };
  
  const handleSubmitAnswer = async () => {
    try {
      setLoading(true);
      setError(null);
      let answerText = answer; 

      // If audio exists, transcribe it and use it as the answer
      if (audioBlob) {
        try {
          const transcript = await ApiClient.transcribeAudio(audioBlob);
          answerText = transcript; 
        } catch (transcriptionError) {
           console.error("Transcription Error:", transcriptionError);
           setError(`Failed to transcribe audio: ${transcriptionError.message}. Using typed answer if available.`);
           if (!answerText) {
                setLoading(false);
                return; 
           }
        }
      }
      
      if (!question || !answerText) {
          setError("Missing question or answer.");
          setLoading(false);
          return;
      }

      // 1. Evaluate Answer
      // Pass expected_skills from the question object
      const evalResponse = await ApiClient.evaluateAnswer(question.content, answerText, question.expected_skills);
      if (!evalResponse || !evalResponse.metrics) {
          throw new Error("Failed to get evaluation metrics from backend.");
      }
      const metrics = evalResponse.metrics;

      // 2. Generate Feedback using metrics
      const feedbackResponse = await ApiClient.generateFeedback(question.content, answerText, metrics);
      if (!feedbackResponse || !feedbackResponse.feedback) {
          throw new Error("Failed to get feedback content from backend.");
      }
      
      const finalFeedback = feedbackResponse.feedback;
      
      if (!finalFeedback.metrics) {
          finalFeedback.metrics = metrics;
      }

      setFeedback(finalFeedback);
      setLoading(false);
      setActiveStep(3); 

      // 3. Navigate to Results Page after feedback is set
       navigate('/results', {
        state: {
          question,
          answer: {
            id: `a_${question.id || 'temp'}`, 
            content: answerText,
            audio_url: audioUrl
          },
          feedback: finalFeedback 
        }
      });

      setActiveStep(3);
    } catch (err) {
      setError('Failed to process answer. Please try again.');
      setLoading(false);
    }
  };
  
  // Handle moving to the next step
  const handleNext = () => {
    if (activeStep === 0 && !selectedRole) {
      setError('Please select a role to continue.');
      return;
    }
    
    if (activeStep === 0) {
      handleGetQuestion();
    } else if (activeStep === 1) {
      setActiveStep(2);
    } else if (activeStep === 2) {
      if (!answer && !audioUrl) {
        setError('Please provide an answer or record audio to continue.');
        return;
      }
      handleSubmitAnswer();
    } else {
      // Navigation is now handled within handleSubmitAnswer after feedback is received
    }
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  const getRandomQuestion = (role, type) => {
    const technicalQuestions = {
      'Software Engineer': [
        'Explain the difference between depth-first search and breadth-first search algorithms and when you would use each.',
        'What are the principles of object-oriented programming and how have you applied them in your projects?',
        'Explain the concept of time and space complexity and provide an example of an O(n log n) algorithm.'
      ],
      'Data Scientist': [
        'Describe how you would handle imbalanced datasets in a classification problem.',
        'Explain the difference between supervised and unsupervised learning with examples.',
        'How would you deal with missing data in a dataset? Explain different approaches and their trade-offs.'
      ],
      'Default': [
        `What are the key technical skills required for a ${role} position?`,
        `Describe a challenging technical problem you solved as a ${role}.`,
        `How do you stay updated with the latest technologies and methodologies in the ${role} field?`
      ]
    };
    
    const behavioralQuestions = {
      'Default': [
        `Describe a challenging project you worked on as a ${role} and how you overcame the obstacles.`,
        `Tell me about a time when you had to work under pressure to meet a deadline as a ${role}.`,
        `Give an example of a time when you had to learn a new technology or methodology quickly for a ${role} project.`
      ]
    };
    
    const questions = type === 'technical' 
      ? (technicalQuestions[role] || technicalQuestions['Default'])
      : (behavioralQuestions[role] || behavioralQuestions['Default']);
    
    return questions[Math.floor(Math.random() * questions.length)];
  };
  
  const getRandomSkills = (role) => {
    const skillsByRole = {
      'Software Engineer': ['Algorithms', 'Data Structures', 'Problem Solving', 'OOP', 'System Design'],
      'Data Scientist': ['Machine Learning', 'Statistics', 'Data Analysis', 'Python', 'SQL'],
      'Default': ['Technical Knowledge', 'Problem Solving', 'Communication', 'Teamwork']
    };
    
    const skills = skillsByRole[role] || skillsByRole['Default'];
    return skills.slice(0, 3); 
  };
  
  const generateMockFeedback = (question, answerText) => {
    return `## Feedback on your answer to: '${question.content}'

### Overall Assessment
Your answer demonstrates good understanding of the topic, but could benefit from more specific examples.

### Strengths
- You've demonstrated strong technical knowledge in ${question.expected_skills[0]}.
- Your explanation is presented in a clear and logical manner.
- Your response is highly relevant to the question asked.

### Areas for Improvement
- Your answer could be more comprehensive by addressing ${question.expected_skills[1]} in more detail.
- Consider using more specific examples to illustrate your points.
- Try organizing your answer with a clearer structure for better clarity.

### Evaluation Metrics
- Technical Accuracy: 0.82
- Completeness: 0.75
- Clarity: 0.88
- Relevance: 0.92
- Overall Score: 0.84

### Role-Specific Advice for ${question.role}
For the ${question.role} role, it's particularly important to demonstrate strong skills in ${question.expected_skills.join(', ')}. Focus on these areas in your interview preparation.`;
  };
  
  // Render step content based on active step
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              Select a role for your interview practice
            </Typography>
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel id="role-select-label">Role</InputLabel>
              <Select
                labelId="role-select-label"
                id="role-select"
                value={selectedRole}
                label="Role"
                onChange={handleRoleChange}
              >
                {roles.map((role) => (
                  <MenuItem key={role} value={role}>{role}</MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel id="type-select-label">Question Type</InputLabel>
              <Select
                labelId="type-select-label"
                id="type-select"
                value={questionType}
                label="Question Type"
                onChange={handleTypeChange}
              >
                <MenuItem value="technical">Technical</MenuItem>
                <MenuItem value="behavioral">Behavioral</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel id="difficulty-select-label">Difficulty</InputLabel>
              <Select
                labelId="difficulty-select-label"
                id="difficulty-select"
                value={difficulty}
                label="Difficulty"
                onChange={handleDifficultyChange}
              >
                <MenuItem value="easy">Easy</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="hard">Hard</MenuItem>
              </Select>
            </FormControl>
          </Box>
        );
      case 1:
        return (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              Your Interview Question
            </Typography>
            <Paper elevation={3} sx={{ p: 3, mt: 2, backgroundColor: '#f8f9fa' }}>
              <Typography variant="h6" gutterBottom>
                {question?.content}
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Role: {question?.role}
                </Typography>
                <Typography variant="subtitle2" color="text.secondary">
                  Type: {question?.type} | Difficulty: {question?.difficulty}
                </Typography>
                <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {question?.expected_skills.map((skill, index) => (
                    <Chip key={index} label={skill} size="small" color="primary" variant="outlined" />
                  ))}
                </Box>
              </Box>
              <Button 
                variant="outlined" 
                startIcon={<VolumeUpIcon />} 
                sx={{ mt: 2 }}
                onClick={handleSpeakQuestion}
              >
                Read Question Aloud
              </Button>
            </Paper>
          </Box>
        );
      case 2:
        return (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              Record Your Answer
            </Typography>
            <Paper elevation={3} sx={{ p: 3, mt: 2, mb: 3, backgroundColor: '#f8f9fa' }}>
              <Typography variant="body1" gutterBottom>
                {question?.content}
              </Typography>
            </Paper>
            
            <Stack direction="column" spacing={3}>
              <Box>
                <Typography variant="h6" gutterBottom>
                  Option 1: Record Audio
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  {!isRecording ? (
                    <Button 
                      variant="contained" 
                      color="primary"
                      startIcon={<MicIcon />}
                      onClick={handleStartRecording}
                      disabled={!!audioUrl}
                    >
                      Start Recording
                    </Button>
                  ) : (
                    <Button 
                      variant="contained" 
                      color="secondary"
                      startIcon={<StopIcon />}
                      onClick={handleStopRecording}
                      className="recording"
                    >
                      Stop Recording
                    </Button>
                  )}
                  
                  {audioUrl && (
                    <Box sx={{ ml: 2 }}>
                      <audio src={audioUrl} controls></audio>
                    </Box>
                  )}
                </Box>
              </Box>
              
              <Box>
                <Typography variant="h6" gutterBottom>
                  Option 2: Type Your Answer
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={6}
                  variant="outlined"
                  placeholder="Type your answer here..."
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                />
              </Box>
            </Stack>
          </Box>
        );
      case 3:
        return (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              Feedback on Your Answer
            </Typography>
            <Paper elevation={3} sx={{ p: 3, mt: 2 }}>
              <Box className="feedback-section" sx={{ whiteSpace: 'pre-line' }}>
                {feedback?.content}
              </Box>
            </Paper>
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };
  
  return (
    <Box sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Practice
      </Typography>
      
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        getStepContent(activeStep)
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          color="inherit"
          disabled={activeStep === 0 || loading}
          onClick={handleBack}
          sx={{ mr: 1 }}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={handleNext}
          disabled={loading}
        >
          {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
        </Button>
      </Box>
    </Box>
  );
};

export default InterviewPage;
