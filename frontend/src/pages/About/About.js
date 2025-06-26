import React from 'react';
import './About.css';

export default function About() {
    return (
        <div className="about-container">
            <h1 className="about-heading">Know the Mission</h1>
            <div className="about-main">
                <div className="about-left">
                    <div className="about-item">
                        <h3 className="section-heading">Project Vision</h3>
                        <p className="about-content">
                            Yoga Pose Detector is a real-time, AI-powered yoga trainer that observes your body posture and provides live feedback.
                            It’s crafted for learners, coders, and wellness enthusiasts.
                        </p>
                    </div>
                    <div className="about-item">
                        <h3 className="section-heading">System Insight</h3>
                        <p className="about-content">
                            Using TensorFlow MoveNet, the app identifies your body keypoints and evaluates your yoga posture with 95%+ accuracy.
                            When a pose is correct, it lights up the skeleton in green. The classification model was fine-tuned using neural networks built on top of MoveNet.
                        </p>
                    </div>
                    <div className="about-item">
                        <h3 className="section-heading">Behind the Scenes</h3>
                        <p className="about-content">
                            The core AI model was trained in Python and exported using TensorFlow.js for seamless use in browsers.
                        </p>
                    </div>
                </div>

                <div className="about-right">
                    <div className="about-item">
                        <h3 className="section-heading">The Creator</h3>
                        <p className="about-content">
                            I’m a full-stack dev and AI hobbyist, passionate about blending wellness with code. My mission is to share tools and insights that inspire others to build, learn, and grow.
                        </p>
                    </div>
                    <div className="about-item">
                        <h3 className="section-heading">Reach Out</h3>
                        <div className="contact-links">
                            <a href="https://www.instagram.com/" target="_blank" rel="noopener noreferrer">
                                <p className="about-content">Instagram</p>
                            </a>
                        </div>
                    </div>
                    <div className="about-item">
                        <h3 className="section-heading">Future Plans</h3>
                        <p className="about-content">
                            Upcoming versions will include pose correction feedback, voice guidance, and session tracking for a better learning experience. Stay tuned for more enhancements!
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}