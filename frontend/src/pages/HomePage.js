import React from 'react';
import { useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import WorkIcon from '@mui/icons-material/Work';
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import FeedbackIcon from '@mui/icons-material/Feedback';
import Paper from '@mui/material/Paper';

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <Box sx={{ py: 4 }}>
      {/* Hero Section */}
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          mb: 4, 
          backgroundColor: '#f0f7ff',
          borderRadius: 2
        }}
      >
        <Typography variant="h2" component="h1" gutterBottom>
          Practice Your Interview Skills with AI
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Our AI-powered interview assistant helps you prepare for job interviews 
          with customized questions, real-time feedback, and detailed evaluations.
        </Typography>
        <Button 
          variant="contained" 
          size="large" 
          sx={{ mt: 2 }}
          onClick={() => navigate('/interview')}
        >
          Start Interview Practice
        </Button>
      </Paper>

      {/* Features Section */}
      <Typography variant="h3" gutterBottom sx={{ mb: 3 }}>
        How It Works
      </Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <WorkIcon color="primary" sx={{ fontSize: 40, mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Role-Specific Questions
              </Typography>
              <Typography variant="body1">
                Select from 13 different job roles and get tailored technical and behavioral 
                interview questions specific to your career path.
              </Typography>
            </CardContent>
            {/* <CardActions>
              <Button size="small" color="primary">
                Browse Roles
              </Button>
            </CardActions> */}
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <RecordVoiceOverIcon color="primary" sx={{ fontSize: 40, mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Speech Recognition
              </Typography>
              <Typography variant="body1">
                Practice answering out loud! Our system converts your spoken answers to text
                for evaluation, just like in a real interview setting.
              </Typography>
            </CardContent>
            {/* <CardActions>
              <Button size="small" color="primary">
                Learn More
              </Button>
            </CardActions> */}
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <FeedbackIcon color="primary" sx={{ fontSize: 40, mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                AI-Powered Feedback
              </Typography>
              <Typography variant="body1">
                Receive detailed feedback on technical accuracy, completeness, clarity, and relevance
                of your answers, with specific suggestions for improvement.
              </Typography>
            </CardContent>
            {/* <CardActions>
              <Button size="small" color="primary">
                View Sample
              </Button>
            </CardActions> */}
          </Card>
        </Grid>
      </Grid>

      {/* CTA Section */}
      <Box sx={{ textAlign: 'center', mt: 6, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Ready to Ace Your Next Interview?
        </Typography>
        <Button 
          variant="contained" 
          color="primary"

          size="large"
          onClick={() => navigate('/interview')}
          sx={{ mt: 2 }}
        >
          Start Practicing Now
        </Button>
      </Box>
    </Box>
  );
};

export default HomePage;
