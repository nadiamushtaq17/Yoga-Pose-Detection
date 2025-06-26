import React from 'react'
import { Link } from 'react-router-dom'

import './Home.css'

export default function Home() {
    return (
        <div className='home-container'>
            <div className='home-header'>
                <h1 className='home-heading'>Yoga Pose Detector</h1>
                <Link to='/about'>
                    <button 
                        className="btn btn-secondary" 
                        id="about-btn"
                    >
                        Know Us
                    </button>
                </Link>
            </div>

            <h1 className="description">Your Smart Yoga Guide</h1>
            <div className="home-main">
                <div className="btn-section">
                    <Link to='/start'>
                        <button className="btn start-btn">
                            Begin Journey
                        </button>
                    </Link>
                    <Link to='/tutorials'>
                        <button className="btn start-btn">
                            Learn Moves
                        </button>
                    </Link>
                </div>
            </div>
        </div>
    )
}