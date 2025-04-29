import React, { useEffect, useRef } from 'react'; // Added useRef
import { useLocation, useNavigate } from 'react-router-dom';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import Divider from '@mui/material/Divider';
import Grid from '@mui/material/Grid';
import LinearProgress from '@mui/material/LinearProgress';
import Stack from '@mui/material/Stack';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import ReactMarkdown from 'react-markdown';
import ReplayIcon from '@mui/icons-material/Replay';
import SaveIcon from '@mui/icons-material/Save';
import HomeIcon from '@mui/icons-material/Home';
import Chip from '@mui/material/Chip';

// Progress bar with label component
const LinearProgressWithLabel = ({ value, label }) => {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
      <Box sx={{ width: '40%', mr: 1 }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
      </Box>
      <Box sx={{ width: '45%', mr: 1 }}>
        <LinearProgress 
          variant="determinate" 
          value={value * 100} 
          sx={{ 
            height: 10, 
            borderRadius: 5,
            backgroundColor: '#e0e0e0',
            '& .MuiLinearProgress-bar': {
              backgroundColor: getColorForScore(value),
            },
          }} 
        />
      </Box>
      <Box sx={{ width: '15%' }}>
        <Typography variant="body2" color="text.secondary">
          {Math.round(value * 100)}%
        </Typography>
      </Box>
    </Box>
  );
};

// Helper function to get color based on score
const getColorForScore = (score) => {
  if (score >= 0.8) return '#4caf50'; // green
  if (score >= 0.6) return '#8bc34a'; // light green
  if (score >= 0.4) return '#ffc107'; // amber
  if (score >= 0.2) return '#ff9800'; // orange
  return '#f44336'; // red
};

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const resultsRef = useRef(null); 
  
  // Get the question, answer, and feedback from location state
  const { question, answer, feedback } = location.state || {};
  
  useEffect(() => {
    if (!question || !feedback) {
      navigate('/interview');
    }
  }, [navigate, question, feedback]);
  
  if (!question || !feedback) {
    return null;
  }
  
  // Handle starting a new interview
  const handleNewInterview = () => {
    navigate('/interview');
  };
  
  // Handle saving results as PDF
  const handleSaveResults = () => {
    const input = resultsRef.current;
    if (!input) {
      console.error("Results area ref not found!");
      alert("Could not find the results content to save.");
      return;
    }

    console.log("Starting PDF generation...");
    input.classList.add('pdf-capture-active');

    html2canvas(input, {
      scale: 2.5, 
      useCORS: true,
      logging: true,
      windowHeight: input.scrollHeight,
      scrollY: -window.scrollY
    })
    .then((canvas) => {
      console.log("Canvas generated, width:", canvas.width, "height:", canvas.height);
      input.classList.remove('pdf-capture-active');
      const imgData = canvas.toDataURL('image/png');
      
      // Use standard A4 size in points (pt)
      const pdf = new jsPDF({
        orientation: 'p',
        unit: 'pt',
        format: 'a4'
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      console.log("PDF Page dimensions (pt):", pageWidth, "x", pageHeight);

      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      
      // Calculate the image height when scaled to fit the PDF page width
      const ratio = imgWidth / imgHeight;
      const pdfImageWidth = pageWidth;
      const pdfImageHeight = pdfImageWidth / ratio;
      console.log("Scaled image dimensions (pt):", pdfImageWidth, "x", pdfImageHeight);

      // Calculate the number of pages needed
      const totalPages = Math.ceil(pdfImageHeight / pageHeight);
      console.log("Total pages needed:", totalPages);

      // Add the image to potentially multiple pages
      for (let i = 0; i < totalPages; i++) {
        if (i > 0) {
          pdf.addPage();
        }
        // Calculate the Y position to shift the image up on subsequent pages
        const yPos = -(pageHeight * i);
        console.log(`Adding image to page ${i + 1} at yPos: ${yPos}`);
        pdf.addImage(imgData, 'PNG', 0, yPos, pdfImageWidth, pdfImageHeight);
      }

      pdf.save(`PrepIQ-Interview-Results-${new Date().toISOString().slice(0, 10)}.pdf`);
      console.log("PDF saved successfully.");

    })
    .catch(err => {
      input.classList.remove('pdf-capture-active'); 
      console.error("Error generating PDF:", err);
      alert("Failed to generate PDF. Check the console for details.");
    });
  };
  
  
  return (
    <Box sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Interview Results
      </Typography>
      
      {/* Add the ref to the main container we want to capture */}
      <Grid container spacing={4} ref={resultsRef}>
        {/* Question Section */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Question:
              </Typography>
              <Typography variant="body1" paragraph>
                {question.content}
              </Typography>
              
              <Typography variant="subtitle2" color="text.secondary">
                Role: {question.role}
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                Type: {question.type} | Difficulty: {question.difficulty}
              </Typography>
              
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {question.expected_skills.map((skill, index) => (
                  <Chip key={index} label={skill} size="small" color="primary" variant="outlined" />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Answer Section */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Your Answer:
              </Typography>
              <Typography variant="body1" paragraph sx={{ minHeight: 100 }}>
                {answer.content || "Audio response recorded"}
              </Typography>
              
              {answer.audio_url && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Audio Recording:
                  </Typography>
                  <audio src={answer.audio_url} controls></audio>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Feedback Section */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3, mt: 2 }}>
            <Typography variant="h5" gutterBottom>
              Feedback & Evaluation
            </Typography>
            
            <Grid container spacing={4} sx={{ display: 'flex', justifyContent: 'center' }}> {/* Added centering styles */}
              <Grid item xs={12} md={7}>
                <Box sx={{ mt: 2 }}>
                  <ReactMarkdown>
                    {feedback.content}
                  </ReactMarkdown>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={5}>
                <Box sx={{ p: 3, bgcolor: '#f5f5f5', borderRadius: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Performance Metrics
                  </Typography>
                  
                  <Box sx={{ mt: 3 }}>
                    <LinearProgressWithLabel 
                      value={feedback.metrics.technical_accuracy} 
                      label="Technical Accuracy" 
                    />
                    <LinearProgressWithLabel 
                      value={feedback.metrics.completeness} 
                      label="Completeness" 
                    />
                    <LinearProgressWithLabel 
                      value={feedback.metrics.clarity} 
                      label="Clarity" 
                    />
                    <LinearProgressWithLabel 
                      value={feedback.metrics.relevance} 
                      label="Relevance" 
                    />
                    
                    <Divider sx={{ my: 2 }} />
                    
                    <LinearProgressWithLabel 
                      value={feedback.metrics.overall_score} 
                      label="Overall Score" 
                    />
                  </Box>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
      
      {/* Action Buttons */}
      <Stack 
        direction={{ xs: 'column', sm: 'row' }} 
        spacing={2} 
        justifyContent="center" 
        sx={{ mt: 4 }}
      >
        <Button 
          variant="contained" 
          color="primary" 
          startIcon={<ReplayIcon />}
          onClick={handleNewInterview}
        >
          Practice Another Question
        </Button>
        <Button 
          variant="outlined" 
          startIcon={<SaveIcon />}
          onClick={handleSaveResults}
        >
          Save Results
        </Button>
        {/* Removed Share Results Button */}
        <Button
          variant="outlined"
          startIcon={<HomeIcon />}
          onClick={() => navigate('/')}
        >
          Back to Home
        </Button>
      </Stack>
    </Box>
  );
};

export default ResultsPage;
